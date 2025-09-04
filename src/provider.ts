import * as vscode from 'vscode';
import {
    CancellationToken,
    LanguageModelChatInformation,
    LanguageModelChatMessage,
    LanguageModelChatProvider,
    LanguageModelChatRequestHandleOptions,
    LanguageModelResponsePart,
    Progress,
} from 'vscode';

/*
 * DEBUG LOGGING FOR TOOL CALLING:
 * This file includes comprehensive debug logging prefixed with '[HF Debug]' to help troubleshoot tool calling issues.
 * Look for these log messages in the VS Code developer console (Help > Toggle Developer Tools > Console):
 *
 * - Message conversion: Shows how VS Code messages are converted to OpenAI format
 * - Tool conversion: Shows how VS Code tools are converted to OpenAI function definitions
 * - Request validation: Validates message structure and tool call/result pairing
 * - API requests: Shows the full request body sent to Hugging Face
 * - API responses: Shows streaming response chunks and their processing
 * - Tool emissions: Shows when tool calls are emitted to VS Code
 *
 * Common issues to look for:
 * - Messages not being converted properly (check roles and content parts)
 * - Tools not being included in requests (check tool conversion)
 * - Invalid tool names (must match ^[\w-]+$)
 * - Missing tool results for tool calls (validation errors)
 * - HF API responses not containing expected tool_call structures
 * - JSON parsing errors in streaming responses
 */

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

interface OpenAIToolCall { id: string; type: 'function'; function: { name: string; arguments: string } };
interface OpenAIFunctionToolDef { type: 'function'; function: { name: string; description?: string; parameters?: object } };

/* ================================================================================================
 * Provider: unchanged API surface; wires the wrapper in.
 * ==============================================================================================*/
export class HuggingFaceChatModelProvider implements LanguageModelChatProvider {
    private readonly _lmWrapper: HuggingFaceLanguageModelWrapper;
    private _chatEndpoints: { model: string; modelMaxPromptTokens: number }[] = [];

    constructor(private readonly secrets: vscode.SecretStorage) {
        this._lmWrapper = new HuggingFaceLanguageModelWrapper(this.secrets);
    }


    async prepareLanguageModelChatInformation(options: { silent: boolean; }, _token: CancellationToken): Promise<LanguageModelChatInformation[]> {
        const apiKey = await this.ensureApiKey(options.silent);
        if (!apiKey) {
            return [];
        }

        const { models, hfInfoMap } = await this.fetchModels(apiKey);

        const infos: LanguageModelChatInformation[] = models.map((m) => {
            const contextLen = m?.providers?.[0]?.context_length ?? DEFAULT_CONTEXT_LENGTH;
            const maxOutput = DEFAULT_MAX_OUTPUT_TOKENS;
            const maxInput = Math.max(1, contextLen - maxOutput);
            const toolCalling = !!(m?.providers?.some(p => p.supports_tools === true));
            const hfInfo = hfInfoMap.get(m.id);
            const vision = hfInfo?.pipeline_tag === 'image-text-to-text';

            return {
                id: m.id,
                name: m.id,
                tooltip: 'Hugging Face',
                family: 'huggingface',
                version: '1.0.0',
                maxInputTokens: maxInput,
                maxOutputTokens: maxOutput,
                capabilities: {
                    toolCalling,
                    imageInput: vision
                }
            } satisfies LanguageModelChatInformation;
        });

        this._chatEndpoints = infos.map(info => ({
            model: info.id,
            modelMaxPromptTokens: info.maxInputTokens + info.maxOutputTokens
        }));

        return infos;
    }

    private async fetchModels(apiKey: string): Promise<{ models: HFModelItem[]; hfInfoMap: Map<string, HFExtraModelInfo> }> {
        const modelsList = (async () => {
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
            return parsed.data ?? [];
        })();

        const ModelsInfo = (async () => {
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
        })();

        const [models, infos] = await Promise.all([modelsList, ModelsInfo]);
        return { models, hfInfoMap: infos };
    }

    async provideLanguageModelChatResponse(
        model: LanguageModelChatInformation,
        messages: readonly LanguageModelChatMessage[],
        options: LanguageModelChatRequestHandleOptions,
        progress: Progress<LanguageModelResponsePart>,
        token: CancellationToken
    ): Promise<void> {
        return this._lmWrapper.provideLanguageModelChatResponse(model, messages, options, progress, token);
    }

    async provideTokenCount(
        model: LanguageModelChatInformation,
        text: string | LanguageModelChatMessage,
        _token: CancellationToken
    ): Promise<number> {
        return this._lmWrapper.provideTokenCount(model, text);
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
}

/* ================================================================================================
 * Wrapper: mirrors Copilot logic (validation → budgeting → safety prompt → request → streaming).
 * ==============================================================================================*/
class HuggingFaceLanguageModelWrapper {
    constructor(private readonly secrets: vscode.SecretStorage) {}

    // Buffers streaming tool_calls arguments by index until complete JSON is assembled
    private _toolCallBuffers: Map<number, { id?: string; name?: string; args: string }> = new Map<number, { id?: string; name?: string; args: string }>();

    async provideLanguageModelChatResponse(
        model: vscode.LanguageModelChatInformation,
        messages: readonly vscode.LanguageModelChatMessage[],
        options: vscode.LanguageModelChatRequestHandleOptions,
        progress: vscode.Progress<vscode.LanguageModelResponsePart>,
        token: vscode.CancellationToken
    ): Promise<void> {
        console.log('[HF Debug] Starting chat response for model:', model.id);
        console.log('[HF Debug] Input messages:', messages.length);
        console.log('[HF Debug] Tool options:', options.tools ? options.tools.length : 0, 'tools');

        const apiKey = await this.ensureApiKey();
        if (!apiKey) {
            throw new Error('Hugging Face API key not found');
        }

        // Convert messages to OpenAI format
        const openaiMessages = this.convertMessages(messages);
        console.log('[HF Debug] Converted messages:', openaiMessages.length);

        // Validate the request structure
        this.validateRequest(messages);

        // Convert tools if present
        const toolConfig = this.convertTools(options);
        console.log('[HF Debug] Tool config:', {
            hasTools: !!toolConfig.tools,
            toolCount: toolConfig.tools?.length || 0,
            toolChoice: toolConfig.tool_choice
        });

        // Validate tools if present
        if (options.tools && options.tools.length > 0) {
            this.validateTools(options.tools);
        }

        // Match Copilot behavior: hard-limit number of tools
        if (options.tools && options.tools.length > 128) {
            throw new Error('Cannot have more than 128 tools per request.');
        }

        // Prepare request body
        const requestBody: Record<string, unknown> = {
            model: model.id,
            messages: openaiMessages,
            stream: true,
            max_tokens: options.modelOptions?.max_tokens || 4096,
            temperature: options.modelOptions?.temperature ?? 0.7,
        };

        // Allow-list model options similar to Copilot's LanguageModelOptions
        if (options.modelOptions) {
            const mo = options.modelOptions as Record<string, unknown>;
            if (typeof mo.stop === 'string' || Array.isArray(mo.stop)) {
                requestBody.stop = mo.stop;
            }
            if (typeof mo.frequency_penalty === 'number') {
                requestBody.frequency_penalty = mo.frequency_penalty;
            }
            if (typeof mo.presence_penalty === 'number') {
                requestBody.presence_penalty = mo.presence_penalty;
            }
        }

        if (toolConfig.tools) {
            requestBody.tools = toolConfig.tools;
        }
        if (toolConfig.tool_choice) {
            requestBody.tool_choice = toolConfig.tool_choice;
        }

        console.log('[HF Debug] Request body:', JSON.stringify(requestBody, null, 2));

        try {
            const response = await fetch(`${BASE_URL}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            console.log('[HF Debug] HF API response status:', response.status);

            if (!response.ok) {
                const errorText = await this.safeReadText(response);
                console.error('[HF Debug] HF API error response:', errorText);
                throw new Error(`Hugging Face API error: ${response.status} ${response.statusText}${errorText ? `\n${errorText}` : ''}`);
            }

            if (!response.body) {
                throw new Error('No response body from Hugging Face API');
            }

            await this.processStreamingResponse(response.body, progress, token);

        } catch (error) {
            console.error('[HF Debug] Error in provideLanguageModelChatResponse:', error);
            throw error;
        }
    }

    async provideTokenCount(
        model: vscode.LanguageModelChatInformation,
        message: string | vscode.LanguageModelChatMessage
    ): Promise<number> {
        if (typeof message === 'string') {
            // Rough estimation: ~4 characters per token
            return Math.ceil(message.length / 4);
        } else {
            // Count tokens in message parts
            let totalTokens = 0;
            for (const part of message.content) {
                if (part instanceof vscode.LanguageModelTextPart) {
                    totalTokens += Math.ceil(part.value.length / 4);
                }
            }
            return totalTokens;
        }
    }

    private async ensureApiKey(): Promise<string | undefined> {
        return await this.secrets.get('huggingface.apiKey');
    }

    private async safeReadText(resp: Response): Promise<string> {
        try {
            return await resp.text();
        } catch {
            return '';
        }
    }

    private async processStreamingResponse(
        responseBody: ReadableStream<Uint8Array>,
        progress: vscode.Progress<vscode.LanguageModelResponsePart>,
        token: vscode.CancellationToken
    ): Promise<void> {
        console.log('[HF Debug] Starting streaming response processing');
        const reader = responseBody.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (!token.isCancellationRequested) {
                const { done, value } = await reader.read();
                if (done) {
                    console.log('[HF Debug] Streaming response complete');
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep incomplete line in buffer

                for (const line of lines) {
                    if (!line.startsWith('data: ')) {
                        continue;
                    }
                    const data = line.slice(6);
                    if (data === '[DONE]') {
                        console.log('[HF Debug] Received [DONE] marker');
                        // Flush any buffered tool_calls on stream end
                        await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ true);
                        continue;
                    }

                    try {
                        const parsed = JSON.parse(data);
                        console.log('[HF Debug] Processing chunk:', {
                            hasChoices: !!parsed.choices,
                            choiceCount: parsed.choices?.length || 0
                        });
                        await this.processDelta(parsed, progress);
                    } catch (parseError) {
                        console.warn('[HF Debug] Failed to parse SSE chunk:', parseError, 'Raw data:', data.substring(0, 200));
                    }
                }
            }
        } finally {
            reader.releaseLock();
            console.log('[HF Debug] Streaming response reader released');
        }
    }

    private async processDelta(delta: Record<string, unknown>, progress: vscode.Progress<vscode.LanguageModelResponsePart>): Promise<void> {
        const choice = (delta.choices as Record<string, unknown>[] | undefined)?.[0];
        if (!choice) {
            console.log('[HF Debug] No choices in delta');
            return;
        }

        // Handle text content
        const deltaObj = choice.delta as Record<string, unknown> | undefined;
        if (deltaObj?.content) {
            const content = deltaObj.content as string;
            console.log('[HF Debug] Processing text delta:', content.substring(0, 100) + (content.length > 100 ? '...' : ''));
            progress.report(new vscode.LanguageModelTextPart(content));
        }

        // Handle tool calls
        if (deltaObj?.tool_calls) {
            const toolCalls = deltaObj.tool_calls as Array<Record<string, unknown>>;
            console.log('[HF Debug] Processing tool calls:', toolCalls.length);

            for (const tc of toolCalls) {
                const idx = (tc.index as number) ?? 0;
                const buf = this._toolCallBuffers.get(idx) ?? { args: '' };
                if (tc.id && typeof tc.id === 'string') {
                    buf.id = tc.id as string;
                }
                const func = tc.function as Record<string, unknown> | undefined;
                if (func?.name && typeof func.name === 'string') {
                    buf.name = func.name as string;
                }
                if (typeof func?.arguments === 'string') {
                    buf.args += func.arguments as string;
                }
                this._toolCallBuffers.set(idx, buf);

                // Try to emit if we already have a complete call
                await this.tryEmitBufferedToolCall(idx, progress);
            }
        } else {
            console.log('[HF Debug] No tool calls in this delta');
        }

        // Some providers set finish_reason to 'tool_calls' when tool calls are complete
        const finish = (choice.finish_reason as string | undefined) ?? undefined;
        if (finish === 'tool_calls' || finish === 'stop') {
            await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ finish === 'tool_calls');
        }
    }

    private async tryEmitBufferedToolCall(index: number, progress: vscode.Progress<vscode.LanguageModelResponsePart>): Promise<void> {
        const buf = this._toolCallBuffers.get(index);
        if (!buf) { return; }
        if (!buf.name) { return; }
        // Prefer to wait for id, but generate one if arguments look complete JSON
        const canParse = this.tryParseJSONObject(buf.args);
        if (!canParse.ok) { return; }
        const id = buf.id ?? `call_${Math.random().toString(36).slice(2, 10)}`;
        const parameters = canParse.value;
        progress.report(new vscode.LanguageModelToolCallPart(id, buf.name, parameters));
        this._toolCallBuffers.delete(index);
        console.log('[HF Debug] Emitted buffered tool call:', { index, id, name: buf.name });
    }

    private async flushToolCallBuffers(progress: vscode.Progress<vscode.LanguageModelResponsePart>, throwOnInvalid: boolean): Promise<void> {
        if (this._toolCallBuffers.size === 0) { return; }
        for (const [idx, buf] of Array.from(this._toolCallBuffers.entries())) {
            const parsed = this.tryParseJSONObject(buf.args);
            if (!parsed.ok) {
                if (throwOnInvalid) {
                    console.error('[HF Debug] Final tool call arguments are not valid JSON:', { idx, snippet: (buf.args || '').slice(0, 200) });
                    throw new Error('Invalid JSON for tool call');
                }
                continue;
            }
            const id = buf.id ?? `call_${Math.random().toString(36).slice(2, 10)}`;
            const name = buf.name ?? 'unknown_tool';
            progress.report(new vscode.LanguageModelToolCallPart(id, name, parsed.value));
            console.log('[HF Debug] Flushed buffered tool call:', { idx, id, name });
            this._toolCallBuffers.delete(idx);
        }
    }

    private tryParseJSONObject(text: string): { ok: true; value: Record<string, unknown> } | { ok: false } {
        try {
            // Must be non-empty and start with { or [ to be valid JSON object/array
            if (!text || !/[{]/.test(text)) {
                return { ok: false };
            }
            const value = JSON.parse(text);
            if (value && typeof value === 'object' && !Array.isArray(value)) {
                return { ok: true, value };
            }
            return { ok: false };
        } catch {
            return { ok: false };
        }
    }

    private convertMessages(messages: readonly vscode.LanguageModelChatMessage[]): Array<{ role: 'system' | 'user' | 'assistant' | 'tool'; content?: string; name?: string; tool_calls?: OpenAIToolCall[]; tool_call_id?: string }> {
        console.log('[HF Debug] Converting messages, input count:', messages.length);
        const out: Array<{ role: 'system' | 'user' | 'assistant' | 'tool'; content?: string; name?: string; tool_calls?: OpenAIToolCall[]; tool_call_id?: string }> = [];

        for (const m of messages) {
            let role: 'system' | 'user' | 'assistant' | undefined;
            const r = m.role as unknown as number;
            const USER = vscode.LanguageModelChatMessageRole.User as unknown as number;
            const ASSISTANT = vscode.LanguageModelChatMessageRole.Assistant as unknown as number;
            if (r === USER) {
                role = 'user';
            } else if (r === ASSISTANT) {
                role = 'assistant';
            } else {
                // Treat unknown roles (e.g., System in newer APIs) as system
                role = 'system';
            }
            const textParts: string[] = [];
            const toolCalls: OpenAIToolCall[] = [];
            const toolResults: { callId: string; content: string }[] = [];

            console.log('[HF Debug] Processing message role:', m.role, 'content parts:', m.content?.length || 0);

            for (const part of m.content ?? []) {
                if (part instanceof vscode.LanguageModelTextPart) {
                    textParts.push(part.value);
                    console.log('[HF Debug] Found text part:', part.value.substring(0, 100) + (part.value.length > 100 ? '...' : ''));
                } else if (part instanceof vscode.LanguageModelToolCallPart) {
                    const id = part.callId || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
                    let args = '{}';
                    try { args = JSON.stringify(part.input ?? {}); } catch { args = '{}'; }
                    toolCalls.push({ id, type: 'function', function: { name: part.name, arguments: args } });
                    console.log('[HF Debug] Found tool call part:', { id, name: part.name, args: args.substring(0, 200) + (args.length > 200 ? '...' : '') });
                } else if (this.isToolResultPart(part)) {
                    let text = '';
                    const pr = part as { content?: ReadonlyArray<unknown> };
                    for (const c of (pr.content ?? [])) {
                        if (c instanceof vscode.LanguageModelTextPart) {
                            text += c.value;
                        } else if (typeof c === 'string') {
                            text += c;
                        } else {
                            try { text += JSON.stringify(c); } catch { /* ignore */ }
                        }
                    }
                    const callId = (part as { callId?: string }).callId ?? '';
                    toolResults.push({ callId, content: text });
                    console.log('[HF Debug] Found tool result part:', { callId, contentLength: text.length });
                }
            }

            let emittedAssistantToolCall = false;
            if (toolCalls.length > 0) {
                // Allow multiple tool calls, matching OpenAI schema
                out.push({ role: 'assistant', content: textParts.join('') || undefined, tool_calls: toolCalls });
                emittedAssistantToolCall = true;
                console.log('[HF Debug] Emitted assistant message with tool calls:', { toolCallCount: toolCalls.length });
            }

            for (const tr of toolResults) {
                out.push({ role: 'tool', tool_call_id: tr.callId, content: tr.content || '' });
                console.log('[HF Debug] Emitted tool result message:', { callId: tr.callId, contentLength: tr.content?.length || 0 });
            }

            const text = textParts.join('');
            if (text && (role === 'system' || role === 'user' || (role === 'assistant' && !emittedAssistantToolCall))) {
                out.push({ role, content: text });
                console.log('[HF Debug] Emitted text message:', { role, contentLength: text.length });
            }
        }

        console.log('[HF Debug] Conversion complete, output messages:', out.length);
        return out;
    }

    private convertTools(options: vscode.LanguageModelChatRequestHandleOptions): { tools?: OpenAIFunctionToolDef[]; tool_choice?: 'auto' | { type: 'function'; function: { name: string } } } {
        const tools = options.tools ?? [];
        console.log('[HF Debug] Converting tools, input count:', tools.length);

        if (!tools || tools.length === 0) {
            console.log('[HF Debug] No tools to convert');
            return {};
        }

        // Hugging Face API limitation: only one tool call per message
        const effectiveTools = tools; // Do not limit here; request validation already caps count globally
        console.log('[HF Debug] Effective tools:', effectiveTools.length);

        const toolDefs: OpenAIFunctionToolDef[] = effectiveTools.map(t => {
            const hasParams = !!(t.inputSchema && typeof t.inputSchema === 'object' && Object.keys(t.inputSchema as Record<string, unknown>).length);
            const def = {
                type: 'function' as const,
                function: {
                    name: t.name,
                    description: t.description,
                    parameters: hasParams ? t.inputSchema : undefined
                }
            };
            console.log('[HF Debug] Converted tool:', { name: t.name, hasDescription: !!t.description, hasSchema: !!t.inputSchema });
            return def;
        });

        let tool_choice: 'auto' | { type: 'function'; function: { name: string } } = 'auto';
        if (options.toolMode === vscode.LanguageModelChatToolMode.Required) {
            if (tools.length !== 1) {
                console.error('[HF Debug] ToolMode.Required but multiple tools:', tools.length);
                throw new Error('LanguageModelChatToolMode.Required is not supported with more than one tool');
            }
            tool_choice = { type: 'function', function: { name: effectiveTools[0].name } };
            console.log('[HF Debug] Set tool_choice to required:', tool_choice);
        } else {
            console.log('[HF Debug] Tool choice remains auto');
        }

        const result = { tools: toolDefs, tool_choice };
        console.log('[HF Debug] Tool conversion complete:', {
            toolCount: toolDefs.length,
            toolChoice: tool_choice
        });
        return result;
    }

    private validateTools(tools: readonly vscode.LanguageModelChatTool[]): void {
        console.log('[HF Debug] Validating tools:', tools.length);
        for (const tool of tools) {
            console.log('[HF Debug] Validating tool:', tool.name);
            if (!tool.name.match(/^[\w-]+$/)) {
                console.error('[HF Debug] Invalid tool name detected:', tool.name);
                throw new Error(`Invalid tool name "${tool.name}": only alphanumeric characters, hyphens, and underscores are allowed.`);
            }
        }
        console.log('[HF Debug] Tool validation passed');
    }

    private validateRequest(messages: readonly vscode.LanguageModelChatMessage[]): void {
        console.log('[HF Debug] Validating request with', messages.length, 'messages');
        const lastMessage = messages[messages.length - 1];
        if (!lastMessage) {
            console.error('[HF Debug] No messages in request');
            throw new Error('Invalid request: no messages.');
        }

        messages.forEach((message, i) => {
            if (message.role === vscode.LanguageModelChatMessageRole.Assistant) {
                const toolCallIds = new Set(
                    message.content
                        .filter(part => part instanceof vscode.LanguageModelToolCallPart)
                        .map(part => (part as vscode.LanguageModelToolCallPart).callId)
                );
                console.log('[HF Debug] Message', i, 'has', toolCallIds.size, 'tool call IDs');
                if (toolCallIds.size === 0) {
                    return;
                }

                let nextMessageIdx = i + 1;
                const errMsg = 'Invalid request: Tool call part must be followed by a User message with a LanguageModelToolResultPart with a matching callId.';
                while (toolCallIds.size > 0) {
                    const nextMessage = messages[nextMessageIdx++];
                    if (!nextMessage || nextMessage.role !== vscode.LanguageModelChatMessageRole.User) {
                        console.error('[HF Debug] Validation failed: missing tool result for call IDs:', Array.from(toolCallIds));
                        throw new Error(errMsg);
                    }

                    nextMessage.content.forEach(part => {
                        if (!this.isToolResultPart(part)) {
                            const ctorName = (Object.getPrototypeOf(part as object) as { constructor?: { name?: string } } | undefined)?.constructor?.name ?? typeof part;
                            console.error('[HF Debug] Validation failed: expected tool result part, got:', ctorName);
                            throw new Error(errMsg);
                        }
                        const callId = (part as { callId: string }).callId;
                        console.log('[HF Debug] Found matching tool result for call ID:', callId);
                        toolCallIds.delete(callId);
                    });
                }
            }
        });
        console.log('[HF Debug] Request validation passed');
    }

    private isToolResultPart(value: unknown): value is { callId: string; content?: ReadonlyArray<unknown> } {
        if (!value || typeof value !== 'object') {
            return false;
        }
        const obj = value as Record<string, unknown>;
        const hasCallId = typeof obj.callId === 'string';
        const hasContent = 'content' in obj;
        return hasCallId && hasContent;
    }
}

/* ------------------------ small util ------------------------ */
