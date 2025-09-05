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

import type {
    HFModelItem,
    HFModelsResponse,
    HFExtraModelInfo,
} from './types';

import {
    convertTools,
    convertMessages,
    tryParseJSONObject,
    validateTools,
    validateRequest
} from './utils';

const BASE_URL = 'https://router.huggingface.co/v1';
const DEFAULT_MAX_OUTPUT_TOKENS = 16000;
const DEFAULT_CONTEXT_LENGTH = 128000;


/**
 * VS Code Chat provider backed by Hugging Face Inference Providers.
 */
export class HuggingFaceChatModelProvider implements LanguageModelChatProvider {
    private _chatEndpoints: { model: string; modelMaxPromptTokens: number }[] = [];
    /** Buffer for assembling streamed tool calls by index. */
    private _toolCallBuffers: Map<number, { id?: string; name?: string; args: string }> = new Map<number, { id?: string; name?: string; args: string }>();

    /**
     * Create a provider using the given secret storage for the API key.
     * @param secrets VS Code secret storage.
     */
    constructor(private readonly secrets: vscode.SecretStorage) {}


    /**
     * Get the list of available language models contributed by this provider
     * @param options Options which specify the calling context of this function
     * @param token A cancellation token which signals if the user cancelled the request or not
     * @returns A promise that resolves to the list of available language models
     */
    async prepareLanguageModelChatInformation(options: { silent: boolean; }, _token: CancellationToken): Promise<LanguageModelChatInformation[]> {
        const apiKey = await this.ensureApiKey(options.silent);
        if (!apiKey) {
            return [];
        }

        const { models, hfInfoMap } = await this.fetchModels(apiKey);

        const infos: LanguageModelChatInformation[] = models.map((m) => {
            const providers = m?.providers ?? [];
            // Select the first provider that explicitly supports tools
            const toolProvider = providers.find(p => p.supports_tools === true);
            const baseProvider = providers[0];

            const chosen = toolProvider ?? baseProvider;
            const contextLen = chosen?.context_length ?? DEFAULT_CONTEXT_LENGTH;
            const maxOutput = DEFAULT_MAX_OUTPUT_TOKENS;
            const maxInput = Math.max(1, contextLen - maxOutput);
            const toolCalling = !!toolProvider; // only true if an explicit tools provider exists
            const hfInfo = hfInfoMap.get(m.id);
            const vision = hfInfo?.pipeline_tag === 'image-text-to-text';

            const id = toolProvider ? `${m.id}:${toolProvider.provider}` : m.id;
            const tooltip = toolProvider ? `Hugging Face via ${toolProvider.provider}` : 'Hugging Face';

            return {
                id,
                name: m.id,
                tooltip,
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

    /**
     * Fetch the list of models and supplementary metadata from Hugging Face.
     * @param apiKey The HF API key used to authenticate.
     */
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

    /**
     * Returns the response for a chat request, passing the results to the progress callback.
     * The {@linkcode LanguageModelChatProvider} must emit the response parts to the progress callback as they are received from the language model.
     * @param model The language model to use
     * @param messages The messages to include in the request
     * @param options Options for the request
     * @param progress The progress to emit the streamed response chunks to
     * @param token A cancellation token for the request
     * @returns A promise that resolves when the response is complete. Results are actually passed to the progress callback.
     */
    async provideLanguageModelChatResponse(
        model: LanguageModelChatInformation,
        messages: readonly LanguageModelChatMessage[],
        options: LanguageModelChatRequestHandleOptions,
        progress: Progress<LanguageModelResponsePart>,
        token: CancellationToken
    ): Promise<void> {
        console.log('Starting chat response for model:', model.id);
        console.log('Input messages:', messages.length);
        console.log('Tool options:', options.tools ? options.tools.length : 0, 'tools');

        const apiKey = await this.ensureApiKey(true);
        if (!apiKey) {
            throw new Error('Hugging Face API key not found');
        }

        // Convert messages to OpenAI format
        const openaiMessages = convertMessages(messages);
        console.log('Converted messages:', openaiMessages.length);

        // Validate the request structure
        validateRequest(messages);

        // Convert tools if present
        const toolConfig = convertTools(options);
        console.log('Tool config:', {
            hasTools: !!toolConfig.tools,
            toolCount: toolConfig.tools?.length || 0,
            toolChoice: toolConfig.tool_choice
        });

        // Validate tools if present
        if (options.tools && options.tools.length > 0) {
            validateTools(options.tools);
        }

        // Copilot parity: limit number of tools
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

        // Allow-list model options
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

        console.log('Request body:', JSON.stringify(requestBody, null, 2));

        const response = await fetch(`${BASE_URL}/chat/completions`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        console.log('HF API response status:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('HF API error response:', errorText);
            throw new Error(`Hugging Face API error: ${response.status} ${response.statusText}${errorText ? `\n${errorText}` : ''}`);
        }

        if (!response.body) {
            throw new Error('No response body from Hugging Face API');
        }

        await this.processStreamingResponse(response.body, progress, token);
    }

    /**
     * Returns the number of tokens for a given text using the model specific tokenizer logic
     * @param model The language model to use
     * @param text The text to count tokens for
     * @param token A cancellation token for the request
     * @returns A promise that resolves to the number of tokens
     */
    async provideTokenCount(
        model: LanguageModelChatInformation,
        text: string | LanguageModelChatMessage,
        _token: CancellationToken
    ): Promise<number> {
        if (typeof text === 'string') {
            return Math.ceil(text.length / 4);
        } else {
            let totalTokens = 0;
            for (const part of text.content) {
                if (part instanceof vscode.LanguageModelTextPart) {
                    totalTokens += Math.ceil(part.value.length / 4);
                }
            }
            return totalTokens;
        }
    }

    /**
     * Ensure an API key exists in SecretStorage, optionally prompting the user when not silent.
     * @param silent If true, do not prompt the user.
     */
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


    /**
     * Read and parse the HF Router streaming (SSE-like) response and report parts.
     * @param responseBody The readable stream body.
     * @param progress Progress reporter for streamed parts.
     * @param token Cancellation token.
     */
    private async processStreamingResponse(
        responseBody: ReadableStream<Uint8Array>,
        progress: vscode.Progress<vscode.LanguageModelResponsePart>,
        token: vscode.CancellationToken
    ): Promise<void> {
        console.log('Starting streaming response processing');
        const reader = responseBody.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (!token.isCancellationRequested) {
                const { done, value } = await reader.read();
                if (done) {
                    console.log(' Streaming response complete');
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) {
                        continue;
                    }
                    const data = line.slice(6);
                    if (data === '[DONE]') {
                        console.log(' Received [DONE] marker');
                        await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ true);
                        continue;
                    }

                    try {
                        const parsed = JSON.parse(data);
                        console.log(' Processing chunk:', {
                            hasChoices: !!parsed.choices,
                            choiceCount: parsed.choices?.length || 0
                        });
                        await this.processDelta(parsed, progress);
                    } catch (parseError) {
                        console.warn(' Failed to parse SSE chunk:', parseError, 'Raw data:', data.substring(0, 200));
                    }
                }
            }
        } finally {
            reader.releaseLock();
            console.log('Streaming response reader released');
        }
    }

    /**
     * Handle a single streamed delta chunk, emitting text and tool call parts.
     * @param delta Parsed SSE chunk from the Router.
     * @param progress Progress reporter for parts.
     */
    private async processDelta(delta: Record<string, unknown>, progress: vscode.Progress<vscode.LanguageModelResponsePart>): Promise<void> {
        const choice = (delta.choices as Record<string, unknown>[] | undefined)?.[0];
        if (!choice) {
            console.log('No choices in delta');
            return;
        }

        const deltaObj = choice.delta as Record<string, unknown> | undefined;
        if (deltaObj?.content) {
            const content = deltaObj.content as string;
            console.log('Processing text delta:', content.substring(0, 100) + (content.length > 100 ? '...' : ''));
            progress.report(new vscode.LanguageModelTextPart(content));
        }

        if (deltaObj?.tool_calls) {
            const toolCalls = deltaObj.tool_calls as Array<Record<string, unknown>>;
            console.log('Processing tool calls:', toolCalls.length);

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

                await this.tryEmitBufferedToolCall(idx, progress);
            }
        } else {
            console.log('No tool calls in this delta');
        }

        const finish = (choice.finish_reason as string | undefined) ?? undefined;
        if (finish === 'tool_calls' || finish === 'stop') {
            await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ finish === 'tool_calls');
        }
    }

    /**
     * Try to emit a buffered tool call when a valid name and JSON arguments are available.
     * @param index The tool call index from the stream.
     * @param progress Progress reporter for parts.
     */
    private async tryEmitBufferedToolCall(index: number, progress: vscode.Progress<vscode.LanguageModelResponsePart>): Promise<void> {
        const buf = this._toolCallBuffers.get(index);
        if (!buf) { return; }
        if (!buf.name) { return; }
        const canParse = tryParseJSONObject(buf.args);
        if (!canParse.ok) { return; }
        const id = buf.id ?? `call_${Math.random().toString(36).slice(2, 10)}`;
        const parameters = canParse.value;
        progress.report(new vscode.LanguageModelToolCallPart(id, buf.name, parameters));
        this._toolCallBuffers.delete(index);
        console.log('Emitted buffered tool call:', { index, id, name: buf.name });
    }

    /**
     * Flush all buffered tool calls, optionally throwing if arguments are not valid JSON.
     * @param progress Progress reporter for parts.
     * @param throwOnInvalid If true, throw when a tool call has invalid JSON args.
     */
    private async flushToolCallBuffers(progress: vscode.Progress<vscode.LanguageModelResponsePart>, throwOnInvalid: boolean): Promise<void> {
        if (this._toolCallBuffers.size === 0) { return; }
        for (const [idx, buf] of Array.from(this._toolCallBuffers.entries())) {
            const parsed = tryParseJSONObject(buf.args);
            if (!parsed.ok) {
                if (throwOnInvalid) {
                    console.error('Final tool call arguments are not valid JSON:', { idx, snippet: (buf.args || '').slice(0, 200) });
                    throw new Error('Invalid JSON for tool call');
                }
                continue;
            }
            const id = buf.id ?? `call_${Math.random().toString(36).slice(2, 10)}`;
            const name = buf.name ?? 'unknown_tool';
            progress.report(new vscode.LanguageModelToolCallPart(id, name, parsed.value));
            console.log('Flushed buffered tool call:', { idx, id, name });
            this._toolCallBuffers.delete(idx);
        }
    }
}
