/**
 * OpenAI function-call entry emitted by assistant messages.
 */
export interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

/**
 * OpenAI function tool definition used to advertise tools.
 */
export interface OpenAIFunctionToolDef {
  type: 'function';
  function: { name: string; description?: string; parameters?: object };
}

/**
 * OpenAI-style chat message used for router requests.
 */
export interface OpenAIChatMessage {
  role: OpenAIChatRole;
  content?: string;
  name?: string;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

/**
 * A single underlying provider (e.g., together, groq) for a model.
 */
export interface HFProvider {
  provider: string;
  status: string;
  supports_tools?: boolean;
  supports_structured_output?: boolean;
  context_length?: number;
}

/**
 * A model entry returned by the Hugging Face router models endpoint.
 */
export interface HFModelItem {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  providers: HFProvider[];
}

/**
 * Supplemental model info from the Hugging Face hub API.
 */
export interface HFExtraModelInfo {
  id: string;
  pipeline_tag?: string;
}

/**
 * Response envelope for the router models listing.
 */
export interface HFModelsResponse {
  object: string;
  data: HFModelItem[];
}

/**
 * Buffer used to accumulate streamed tool call parts until arguments are valid JSON.
 */
export interface ToolCallBuffer { id?: string; name?: string; args: string }

/** OpenAI-style chat roles. */
export type OpenAIChatRole = 'system' | 'user' | 'assistant' | 'tool';