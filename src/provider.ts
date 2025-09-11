import * as vscode from "vscode";
import {
	CancellationToken,
	LanguageModelChatInformation,
	LanguageModelChatMessage,
	LanguageModelChatProvider,
	LanguageModelChatRequestHandleOptions,
	LanguageModelResponsePart,
	Progress,
} from "vscode";

import type { HFModelItem, HFModelsResponse } from "./types";

import { convertTools, convertMessages, tryParseJSONObject, validateRequest } from "./utils";

const BASE_URL = "https://router.huggingface.co/v1";
const DEFAULT_MAX_OUTPUT_TOKENS = 16000;
const DEFAULT_CONTEXT_LENGTH = 128000;

/**
 * VS Code Chat provider backed by Hugging Face Inference Providers.
 */
export class HuggingFaceChatModelProvider implements LanguageModelChatProvider {
	private _chatEndpoints: { model: string; modelMaxPromptTokens: number }[] = [];
	/** Buffer for assembling streamed tool calls by index. */
	private _toolCallBuffers: Map<number, { id?: string; name?: string; args: string }> = new Map<
		number,
		{ id?: string; name?: string; args: string }
	>();

	/** Indices for which a tool call has been fully emitted. */
	private _completedToolCallIndices = new Set<number>();

    // No user-facing early announcements; buffer until complete JSON.

	/** Track if we emitted any assistant text before seeing tool calls (SSE-like begin-tool-calls hint). */
	private _hasEmittedAssistantText = false;
	/** Track if we emitted the begin-tool-calls whitespace flush. */
	private _emittedBeginToolCallsHint = false;

	/**
	 * Create a provider using the given secret storage for the API key.
	 * @param secrets VS Code secret storage.
	 */
	constructor(private readonly secrets: vscode.SecretStorage) {}

	/** Roughly estimate tokens for VS Code chat messages (text only). */
	private estimateMessagesTokens(msgs: readonly vscode.LanguageModelChatMessage[]): number {
		let total = 0;
		for (const m of msgs) {
			for (const part of m.content) {
				if (part instanceof vscode.LanguageModelTextPart) {
					total += Math.ceil(part.value.length / 4);
				}
			}
		}
		return total;
	}

	/** Rough token estimate for tool definitions by JSON size. */
	private estimateToolTokens(tools: { type: string; function: { name: string; description?: string; parameters?: object } }[] | undefined): number {
		if (!tools || tools.length === 0) { return 0; }
		try {
			const json = JSON.stringify(tools);
			return Math.ceil(json.length / 4);
		} catch {
			return 0;
		}
	}

	/**
	 * Get the list of available language models contributed by this provider
	 * @param options Options which specify the calling context of this function
	 * @param token A cancellation token which signals if the user cancelled the request or not
	 * @returns A promise that resolves to the list of available language models
	 */
	async prepareLanguageModelChatInformation(
		options: { silent: boolean },
		_token: CancellationToken
	): Promise<LanguageModelChatInformation[]> {
		const apiKey = await this.ensureApiKey(options.silent);
		if (!apiKey) {
			return [];
		}

		const { models } = await this.fetchModels(apiKey);

		const infos: LanguageModelChatInformation[] = models.flatMap((m) => {
			const providers = m?.providers ?? [];
			const modalities = m.architecture?.input_modalities ?? [];
			const vision = Array.isArray(modalities) && modalities.includes("image");

			// Build entries for all providers that support tool calling
			const toolProviders = providers.filter((p) => p.supports_tools === true);
			const entries: LanguageModelChatInformation[] = [];

			for (const p of toolProviders) {
				const contextLen = p?.context_length ?? DEFAULT_CONTEXT_LENGTH;
				const maxOutput = DEFAULT_MAX_OUTPUT_TOKENS;
				const maxInput = Math.max(1, contextLen - maxOutput);
				entries.push({
					id: `${m.id}:${p.provider}`,
					name: m.id,
					tooltip: `Hugging Face via ${p.provider}`,
					family: "huggingface",
					version: "1.0.0",
					maxInputTokens: maxInput,
					maxOutputTokens: maxOutput,
					capabilities: {
						toolCalling: true,
						imageInput: vision,
					},
				} satisfies LanguageModelChatInformation);
			}

			// If no tool-capable providers exist, include a single non-tool entry as fallback
			if (entries.length === 0 && providers.length > 0) {
				const base = providers[0];
				const contextLen = base?.context_length ?? DEFAULT_CONTEXT_LENGTH;
				const maxOutput = DEFAULT_MAX_OUTPUT_TOKENS;
				const maxInput = Math.max(1, contextLen - maxOutput);
				entries.push({
					id: m.id,
					name: m.id,
					tooltip: "Hugging Face",
					family: "huggingface",
					version: "1.0.0",
					maxInputTokens: maxInput,
					maxOutputTokens: maxOutput,
					capabilities: {
						toolCalling: false,
						imageInput: vision,
					},
				} satisfies LanguageModelChatInformation);
			}

			return entries;
		});

		this._chatEndpoints = infos.map((info) => ({
			model: info.id,
			modelMaxPromptTokens: info.maxInputTokens + info.maxOutputTokens,
		}));

		return infos;
	}

	async provideLanguageModelChatInformation(
		options: { silent: boolean },
		_token: CancellationToken
	): Promise<LanguageModelChatInformation[]> {
		return this.prepareLanguageModelChatInformation({ silent: options.silent ?? false }, _token);
	}

	/**
	 * Fetch the list of models and supplementary metadata from Hugging Face.
	 * @param apiKey The HF API key used to authenticate.
	 */
	private async fetchModels(
		apiKey: string
	): Promise<{ models: HFModelItem[] }> {
			const modelsList = (async () => {
				const resp = await fetch(`${BASE_URL}/models`, {
					method: "GET",
					headers: { Authorization: `Bearer ${apiKey}` },
				});
				if (!resp.ok) {
					let text = "";
					try {
						text = await resp.text();
					} catch (error) {
						console.error("Failed to read response text:", error);
					}
					const err = new Error(
						`Failed to fetch Hugging Face models: ${resp.status} ${resp.statusText}${text ? `\n${text}` : ""}`
					);
					console.error("Failed to fetch Hugging Face models:", err);
					throw err;
				}
				const parsed = (await resp.json()) as HFModelsResponse;
				return parsed.data ?? [];
			})();

			try {
				const models = await modelsList;
				return { models };
			} catch (err) {
				console.error("Failed to fetch Hugging Face models:", err);
				throw err;
			}
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
        // minimal logging: avoid verbose info logs

		// Reset tool call state for this request
		this._toolCallBuffers.clear();
		this._completedToolCallIndices.clear();
		this._hasEmittedAssistantText = false;
		this._emittedBeginToolCallsHint = false;


		let requestBody: Record<string, unknown> | undefined;
		try {
			const apiKey = await this.ensureApiKey(true);
			if (!apiKey) {
				throw new Error("Hugging Face API key not found");
			}

			// Convert messages to OpenAI format
            const openaiMessages = convertMessages(messages);

			// Validate the request structure
			validateRequest(messages);

			// Convert tools if present
            const toolConfig = convertTools(options);

		// Copilot parity: limit number of tools
        if (options.tools && options.tools.length > 128) {
            throw new Error("Cannot have more than 128 tools per request.");
        }

            // Preflight token budget (approximate)
            const inputTokenCount = this.estimateMessagesTokens(messages);
            const toolTokenCount = this.estimateToolTokens(toolConfig.tools);
            const tokenLimit = Math.max(1, model.maxInputTokens);
            if (inputTokenCount + toolTokenCount > tokenLimit) {
                console.error("Message exceeds token limit", { total: inputTokenCount + toolTokenCount, tokenLimit });
                throw new Error("Message exceeds token limit.");
            }

            // Prepare request body
            requestBody = {
                model: model.id,
                messages: openaiMessages,
                stream: true,
                max_tokens: Math.min(options.modelOptions?.max_tokens || 4096, model.maxOutputTokens),
                temperature: options.modelOptions?.temperature ?? 0.7,
            };

			// Allow-list model options
			if (options.modelOptions) {
				const mo = options.modelOptions as Record<string, unknown>;
				if (typeof mo.stop === "string" || Array.isArray(mo.stop)) {
					(requestBody as Record<string, unknown>).stop = mo.stop;
				}
				if (typeof mo.frequency_penalty === "number") {
					(requestBody as Record<string, unknown>).frequency_penalty = mo.frequency_penalty;
				}
				if (typeof mo.presence_penalty === "number") {
					(requestBody as Record<string, unknown>).presence_penalty = mo.presence_penalty;
				}
			}

			if (toolConfig.tools) {
				(requestBody as Record<string, unknown>).tools = toolConfig.tools;
			}
			if (toolConfig.tool_choice) {
				(requestBody as Record<string, unknown>).tool_choice = toolConfig.tool_choice;
			}

            const response = await fetch(`${BASE_URL}/chat/completions`, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${apiKey}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(requestBody),
            });

			if (!response.ok) {
				const errorText = await response.text();
				console.error("HF API error response:", errorText);
				throw new Error(
					`Hugging Face API error: ${response.status} ${response.statusText}${errorText ? `\n${errorText}` : ""}`
				);
			}

			if (!response.body) {
				throw new Error("No response body from Hugging Face API");
			}
			await this.processStreamingResponse(response.body, progress, token);
		} catch (err) {
			console.error("Chat request failed", {
				modelId: model.id,
				messageCount: messages.length,
				error: err instanceof Error ? { name: err.name, message: err.message } : String(err),
			});
			throw err;
		}
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
		if (typeof text === "string") {
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
		let apiKey = await this.secrets.get("huggingface.apiKey");
		if (!apiKey && !silent) {
			const entered = await vscode.window.showInputBox({
				title: "Hugging Face API Key",
				prompt: "Enter your Hugging Face API key",
				ignoreFocusOut: true,
				password: true,
			});
			if (entered && entered.trim()) {
				apiKey = entered.trim();
				await this.secrets.store("huggingface.apiKey", apiKey);
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
        token: vscode.CancellationToken,
    ): Promise<void> {
        const reader = responseBody.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

		try {
			while (!token.isCancellationRequested) {
				const { done, value } = await reader.read();
                if (done) { break; }

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split("\n");
				buffer = lines.pop() || "";

				for (const line of lines) {
					if (!line.startsWith("data: ")) {
						continue;
					}
					const data = line.slice(6);
                    if (data === "[DONE]") {
                        // Do not throw on [DONE]; any incomplete/empty buffers are ignored.
                        await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ false);
                        continue;
                    }

					try {
						const parsed = JSON.parse(data);
                        await this.processDelta(parsed, progress);
                    } catch (parseError) {
                        console.warn("Failed to parse SSE chunk", {
                            error: parseError instanceof Error ? parseError.message : String(parseError),
                            dataSnippet: data.slice(0, 200),
                        });
                    }
                }
            }
        } finally {
            reader.releaseLock();
            // Clean up any leftover tool call state
            this._toolCallBuffers.clear();
            this._completedToolCallIndices.clear();
            this._hasEmittedAssistantText = false;
            this._emittedBeginToolCallsHint = false;
        }
    }

	/**
	 * Handle a single streamed delta chunk, emitting text and tool call parts.
	 * @param delta Parsed SSE chunk from the Router.
	 * @param progress Progress reporter for parts.
	 */
	private async processDelta(
		delta: Record<string, unknown>,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>
	): Promise<void> {
		const choice = (delta.choices as Record<string, unknown>[] | undefined)?.[0];
        if (!choice) { return; }

		const deltaObj = choice.delta as Record<string, unknown> | undefined;

		// Optional: report thinking progress if backend provides it and host supports it.
		try {
			const maybeThinking = (choice as Record<string, unknown> | undefined)?.thinking ?? (deltaObj as Record<string, unknown> | undefined)?.thinking;
			if (maybeThinking !== undefined) {
				const vsAny = (vscode as unknown as Record<string, unknown>);
				const ThinkingCtor = vsAny["LanguageModelThinkingPart"] as
					| (new (text: string, id?: string, metadata?: unknown) => unknown)
					| undefined;
				if (ThinkingCtor) {
					let text = "";
					let id: string | undefined;
					let metadata: unknown;
					if (maybeThinking && typeof maybeThinking === "object") {
						const mt = maybeThinking as Record<string, unknown>;
						text = typeof mt["text"] === "string" ? (mt["text"] as string) : "";
						id = typeof mt["id"] === "string" ? (mt["id"] as string) : undefined;
						metadata = mt["metadata"];
					} else if (typeof maybeThinking === "string") {
						text = maybeThinking;
					}
					if (text) {
						progress.report(new (ThinkingCtor as new (text: string, id?: string, metadata?: unknown) => unknown)(text, id, metadata) as unknown as vscode.LanguageModelResponsePart);
					}
				}
			}
		} catch {
			// ignore errors; thinking is optional
		}
		if (deltaObj?.content) {
            let content = deltaObj.content as string;
            content = this.stripControlTokens(content);
            if (content.trim().length > 0) {
                progress.report(new vscode.LanguageModelTextPart(content));
                this._hasEmittedAssistantText = true;
            }
		}

			if (deltaObj?.tool_calls) {
                const toolCalls = deltaObj.tool_calls as Array<Record<string, unknown>>;

				// SSEProcessor-like: if first tool call appears after text, emit a whitespace
				// to ensure any UI buffers/linkifiers are flushed without adding visible noise.
				if (!this._emittedBeginToolCallsHint && this._hasEmittedAssistantText && toolCalls.length > 0) {
					progress.report(new vscode.LanguageModelTextPart(" "));
					this._emittedBeginToolCallsHint = true;
				}

				for (const tc of toolCalls) {
					const idx = (tc.index as number) ?? 0;
					// Ignore any further deltas for an index we've already completed
					if (this._completedToolCallIndices.has(idx)) {
						continue;
					}
					const buf = this._toolCallBuffers.get(idx) ?? { args: "" };
					if (tc.id && typeof tc.id === "string") {
						buf.id = tc.id as string;
					}
					const func = tc.function as Record<string, unknown> | undefined;
					if (func?.name && typeof func.name === "string") {
						buf.name = func.name as string;
					}
					if (typeof func?.arguments === "string") {
						buf.args += func.arguments as string;
					}
					this._toolCallBuffers.set(idx, buf);

					// Emit immediately once arguments become valid JSON to avoid perceived hanging
					await this.tryEmitBufferedToolCall(idx, progress);
				}
            }

		const finish = (choice.finish_reason as string | undefined) ?? undefined;
		if (finish === "tool_calls" || finish === "stop") {
			// On both 'tool_calls' and 'stop', emit any buffered calls and throw on invalid JSON
			await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ true);
		}
	}

	/**
	 * Try to emit a buffered tool call when a valid name and JSON arguments are available.
	 * @param index The tool call index from the stream.
	 * @param progress Progress reporter for parts.
	 */
	private async tryEmitBufferedToolCall(
		index: number,
		progress: vscode.Progress<vscode.LanguageModelResponsePart>
	): Promise<void> {
		const buf = this._toolCallBuffers.get(index);
		if (!buf) {
			return;
		}
		if (!buf.name) {
			return;
		}
		const canParse = tryParseJSONObject(buf.args);
		if (!canParse.ok) {
			return;
		}
		const id = buf.id ?? `call_${Math.random().toString(36).slice(2, 10)}`;
		const parameters = canParse.value;
		progress.report(new vscode.LanguageModelToolCallPart(id, buf.name, parameters));
		this._toolCallBuffers.delete(index);
        this._completedToolCallIndices.add(index);
	}

	/**
	 * Flush all buffered tool calls, optionally throwing if arguments are not valid JSON.
	 * @param progress Progress reporter for parts.
	 * @param throwOnInvalid If true, throw when a tool call has invalid JSON args.
	 */
	private async flushToolCallBuffers(
		progress: vscode.Progress<vscode.LanguageModelResponsePart>,
		throwOnInvalid: boolean
	): Promise<void> {
		if (this._toolCallBuffers.size === 0) {
			return;
		}
		for (const [idx, buf] of Array.from(this._toolCallBuffers.entries())) {
			const parsed = tryParseJSONObject(buf.args);
			if (!parsed.ok) {
				if (throwOnInvalid) {
					console.error("Invalid JSON for tool call", { idx, snippet: (buf.args || "").slice(0, 200) });
					throw new Error("Invalid JSON for tool call");
				}
				continue;
			}
			const id = buf.id ?? `call_${Math.random().toString(36).slice(2, 10)}`;
			const name = buf.name ?? "unknown_tool";
            progress.report(new vscode.LanguageModelToolCallPart(id, name, parsed.value));
			this._toolCallBuffers.delete(idx);
			this._completedToolCallIndices.add(idx);
		}
	}

	/** Strip provider control tokens like <|tool_calls_section_begin|> from streamed text. */
	private stripControlTokens(text: string): string {
		try {
			return text.replace(/<\|[a-zA-Z0-9_-]+_section_(begin|end)\|>/g, "");
		} catch {
			return text;
		}
	}

}
