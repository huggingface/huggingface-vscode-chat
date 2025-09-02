import * as vscode from 'vscode';
import {
    CancellationToken,
    LanguageModelChatInformation,
    LanguageModelChatMessage,
    LanguageModelChatMessageRole,
    LanguageModelChatProvider,
    LanguageModelChatRequestHandleOptions,
    LanguageModelResponsePart,
    LanguageModelTextPart,
    Progress
} from 'vscode';

const BASE_URL = 'https://router.huggingface.co/v1';
const DEFAULT_MAX_OUTPUT_TOKENS = 16000;
const DEFAULT_CONTEXT_LENGTH = 128000;

interface HFProvider {
    provider: string;
    status: string;
    supports_tools?: boolean;
    supports_structured_output?: boolean;
    context_length?: number;
}

interface HFModelItem {
    id: string;
    object: string;
    created: number;
    owned_by: string;
    providers: HFProvider[];
}

interface HFExtraModelInfo {
    id: string;
    pipeline_tag?: string;
}

interface HFModelsResponse {
    object: string;
    data: HFModelItem[];
}

interface HFChatChoiceMessage {
    role: string;
    content: string;
    tool_calls?: Array<{
        id: string;
        type: string;
        function: { name: string; arguments: string };
    }>;
}

interface HFChatChoice {
    index: number;
    finish_reason: string | null;
    message: HFChatChoiceMessage;
}

interface HFChatCompletionResponse {
    id: string;
    object: string;
    created: number;
    model: string;
    choices: HFChatChoice[];
}
interface OpenAIToolCall { id: string; type: 'function'; function: { name: string; arguments: string } };
interface OpenAIFunctionToolDef { type: 'function'; function: { name: string; description?: string; parameters?: object } };

// Fetches and composes `LanguageModelChatInformation` for a given model id.
async function getChatModelInfo(
    id: string,
    name: string,
    opts?: { apiKey?: string; routerModel?: HFModelItem; hfInfoMap?: Map<string, HFExtraModelInfo> }
): Promise<LanguageModelChatInformation> {
    let routerModel = opts?.routerModel as HFModelItem | undefined;
    if (!routerModel && opts?.apiKey) {
        try {
            const resp = await fetch(`${BASE_URL}/models`, {
                method: 'GET',
                headers: { 'Authorization': `Bearer ${opts.apiKey}` }
            });
            if (resp.ok) {
                const all = (await resp.json()) as HFModelsResponse;
                routerModel = all.data?.find(m => m.id === id);
            }
        } catch (error) {
            console.error('Failed to fetch router model:', error);
            throw new Error('Unable to fetch model information from Hugging Face.');
        }
    }

    let hfInfo: HFExtraModelInfo | undefined;
    if (opts?.hfInfoMap) {
        hfInfo = opts.hfInfoMap.get(id);
    } else {
        try {
            const infoResp = await fetch('https://huggingface.co/api/models?other=conversational&inference=warm', { method: 'GET' });
            if (infoResp.ok) {
                const infos = (await infoResp.json()) as HFExtraModelInfo[];
                hfInfo = infos.find(i => i.id === id);
            }
        } catch (error) {
            console.error('Failed to fetch optional model metadata:', error);
            // Optional metadata, continue without it
        }
    }

    const contextLen = routerModel?.providers?.[0]?.context_length ?? DEFAULT_CONTEXT_LENGTH;
    const maxOutput = DEFAULT_MAX_OUTPUT_TOKENS;
    const maxInput = Math.max(1, contextLen - maxOutput);
    const toolCalling = !!(routerModel?.providers?.some(p => p.supports_tools === true));
    const vision = hfInfo?.pipeline_tag === 'image-text-to-text';

    return {
        id,
        name,
        tooltip: 'Hugging Face',
        family: 'huggingface',
        version: '1.0.0',
        maxInputTokens: maxInput,
        maxOutputTokens: maxOutput,
        capabilities: {
            toolCalling,
            imageInput: vision
        }
    };
}

export class HuggingFaceChatModelProvider implements LanguageModelChatProvider {
    constructor(private readonly secrets: vscode.SecretStorage) { }

    async prepareLanguageModelChatInformation(options: { silent: boolean; }, _token: CancellationToken): Promise<LanguageModelChatInformation[]> {
        const apiKey = await this.ensureApiKey(options.silent);
        if (!apiKey) {
            return [];
        }

        const models = await this.fetchModels(apiKey);
        const hfInfoMap = await this.fetchHFExtraModelInfoMap();

        const infos = await Promise.all(models.map(async (m) => {
            return getChatModelInfo(m.id, m.id, { apiKey, routerModel: m, hfInfoMap });
        }));

        return infos;
    }

    async provideLanguageModelChatResponse(
        model: LanguageModelChatInformation,
        messages: readonly LanguageModelChatMessage[],
        options: LanguageModelChatRequestHandleOptions,
        progress: Progress<LanguageModelResponsePart>,
        _token: CancellationToken
    ): Promise<void> {
        const apiKey = await this.secrets.get('huggingface.apiKey');
        if (!apiKey) {
            throw new Error('Hugging Face API key not set. Run the Manage command to configure it.');
        }

        const converted = this.convertMessages(messages);
        const toolsPayload = model.capabilities?.toolCalling ? this.convertTools(options) : {};
        const modelOptions = options.modelOptions ?? {};
        const body = {
            model: model.id,
            messages: converted,
            stream: false,
            ...(toolsPayload.tools ? { tools: toolsPayload.tools } : {}),
            ...(toolsPayload.tool_choice ? { tool_choice: toolsPayload.tool_choice } : {}),
            ...modelOptions
        } as const;

        const resp = await fetch(`${BASE_URL}/chat/completions`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(body)
        });

        if (!resp.ok) {
            let text = '';
            try {
                text = await resp.text();
            } catch (error) {
                console.error('Failed to read response text:', error);
            }
            throw new Error(`Hugging Face chat error: ${resp.status} ${resp.statusText}${text ? `\n${text}` : ''}`);
        }

        const data = (await resp.json()) as HFChatCompletionResponse;
        const choice = data.choices?.[0];
        if (!choice) {
            progress.report(new LanguageModelTextPart('[No content]'));
            return;
        }
        const msg = choice.message;
        // Emit tool calls first if present
        if (msg.tool_calls && msg.tool_calls.length > 0) {
            for (const call of msg.tool_calls) {
                const callId = call.id || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
                let argsObj: object = {};
                try { argsObj = call.function?.arguments ? JSON.parse(call.function.arguments) : {}; } catch (error) { console.error('Failed to parse tool call arguments:', error); argsObj = { arguments: call.function?.arguments ?? '' }; }
                progress.report(new vscode.LanguageModelToolCallPart(callId, call.function?.name ?? 'unknown', argsObj));
            }
        }

        const content = msg.content ?? '';
        if (content && content.trim().length > 0) {
            progress.report(new LanguageModelTextPart(content));
        }
    }

    async provideTokenCount(
        model: LanguageModelChatInformation,
        text: string | LanguageModelChatMessage,
        _token: CancellationToken
    ): Promise<number> {
        if (typeof text === 'string') {
            return Math.ceil(text.length / 4);
        }
        const joined = this.extractTextFromMessage(text);
        return Math.ceil(joined.length / 4);
    }

    private async ensureApiKey(silent: boolean): Promise<string | undefined> {
        let apiKey = await this.secrets.get('huggingface.apiKey');
        if (!apiKey && !silent) {
            const entered = await vscode.window.showInputBox({
                title: 'Hugging Face API Key',
                prompt: 'Enter your Hugging Face API key',
                ignoreFocusOut: true,
                password: true
            });
            if (entered && entered.trim()) {
                apiKey = entered.trim();
                await this.secrets.store('huggingface.apiKey', apiKey);
            }
        }
        return apiKey;
    }

    private async fetchModels(apiKey: string): Promise<HFModelItem[]> {
        const resp = await fetch(`${BASE_URL}/models`, {
            method: 'GET',
            headers: { 'Authorization': `Bearer ${apiKey}` }
        });
        if (!resp.ok) {
            let text = '';
            try {
                text = await resp.text();
            } catch (error) {
                console.error('Failed to read response text:', error);
            }
            throw new Error(`Failed to fetch Hugging Face models: ${resp.status} ${resp.statusText}${text ? `\n${text}` : ''}`);
        }
        const parsed = (await resp.json()) as HFModelsResponse;
        // Do not filter by provider status; allow users to pick any
        return parsed.data ?? [];
    }

    private async fetchHFExtraModelInfoMap(): Promise<Map<string, HFExtraModelInfo>> {
        const map = new Map<string, HFExtraModelInfo>();
        try {
            const resp = await fetch('https://huggingface.co/api/models?other=conversational&inference=warm', { method: 'GET' });
            if (!resp.ok) {
                return map;
            }
            const arr = (await resp.json()) as HFExtraModelInfo[];
            for (const item of arr) {
                if (item?.id) {
                    map.set(item.id, item);
                }
            }
        } catch (error) {
            console.error('Failed to fetch Hugging Face model metadata:', error);
        }
        return map;
    }

    private convertMessages(messages: readonly LanguageModelChatMessage[]): Array<
        { role: 'user' | 'assistant' | 'tool'; content?: string; name?: string; tool_calls?: OpenAIToolCall[]; tool_call_id?: string }
    > {
        const out: Array<{ role: 'user' | 'assistant' | 'tool'; content?: string; name?: string; tool_calls?: OpenAIToolCall[]; tool_call_id?: string }> = [];
        for (const m of messages) {
            const role = this.mapRole(m.role);
            const textParts: string[] = [];
            const toolCalls: OpenAIToolCall[] = [];
            const toolResults: { callId: string; content: string }[] = [];

            for (const part of m.content ?? []) {
                if (part instanceof vscode.LanguageModelTextPart) {
                    textParts.push(part.value);
                } else if (part instanceof vscode.LanguageModelToolCallPart) {
                    const id = part.callId || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
                    const args = JSON.stringify(part.input ?? {});
                    toolCalls.push({ id, type: 'function', function: { name: part.name, arguments: args } });
                } else if (part instanceof vscode.LanguageModelToolResultPart) {
                    let text = '';
                    for (const c of part.content ?? []) {
                        if (c instanceof vscode.LanguageModelTextPart) {
                            text += c.value;
                        } else if (typeof c === 'string') {
                            text += c;
                        } else {
                            // best-effort stringify
                            try { text += JSON.stringify(c); } catch (error) { console.error('Failed to stringify content:', error); }
                        }
                    }
                    toolResults.push({ callId: part.callId, content: text });
                }
            }

            let emittedAssistantToolCall = false;
            if (toolCalls.length > 0) {
                out.push({ role: 'assistant', content: textParts.join('') || undefined, tool_calls: toolCalls });
                emittedAssistantToolCall = true;
            }

            // Emit tool results as their own messages (OpenAI-style 'tool' role)
            for (const tr of toolResults) {
                out.push({ role: 'tool', tool_call_id: tr.callId, content: tr.content || '' });
            }

            // Emit regular text-only messages for user/assistant
            const text = textParts.join('');
            if (text && (role === 'user' || (role === 'assistant' && !emittedAssistantToolCall))) {
                out.push({ role, content: text });
            }
        }
        return out;
    }

    private mapRole(role: LanguageModelChatMessageRole): 'user' | 'assistant' | undefined {
        switch (role) {
            case vscode.LanguageModelChatMessageRole.User: return 'user';
            case vscode.LanguageModelChatMessageRole.Assistant: return 'assistant';
            default: return undefined;
        }
    }

    private extractTextFromMessage(m: LanguageModelChatMessage): string {
        const parts = m.content ?? [];
        const texts: string[] = [];
        for (const part of parts) {
            if (part instanceof vscode.LanguageModelTextPart) {
                texts.push(part.value);
            }
        }
        return texts.join('');
    }

    private convertTools(options: LanguageModelChatRequestHandleOptions): { tools?: OpenAIFunctionToolDef[]; tool_choice?: 'auto' | 'required' } {
        const tools = options.tools ?? [];
        if (!tools || tools.length === 0) {
            return {};
        }
        const toolDefs: OpenAIFunctionToolDef[] = tools.map(t => ({
            type: 'function',
            function: {
                name: t.name,
                description: t.description,
                parameters: t.inputSchema ?? { type: 'object', properties: {}, additionalProperties: true }
            }
        }));
        const tool_choice = options.toolMode === vscode.LanguageModelChatToolMode.Required ? 'required' : 'auto';
        return { tools: toolDefs, tool_choice };
    }
}


