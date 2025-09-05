import * as vscode from 'vscode';
import type {
  OpenAIChatMessage,
  OpenAIChatRole,
  OpenAIFunctionToolDef,
  OpenAIToolCall,
} from './types';


/**
 * Convert VS Code chat request messages into OpenAI-compatible message objects.
 * @param messages The VS Code chat messages to convert.
 * @returns OpenAI-compatible messages array.
 */
export function convertMessages(messages: readonly vscode.LanguageModelChatRequestMessage[]): OpenAIChatMessage[] {
  const out: OpenAIChatMessage[] = [];
  for (const m of messages) {
    const role = mapRole(m);
    const textParts: string[] = [];
    const toolCalls: OpenAIToolCall[] = [];
    const toolResults: { callId: string; content: string }[] = [];

    for (const part of m.content ?? []) {
      if (part instanceof vscode.LanguageModelTextPart) {
        textParts.push(part.value);
      } else if (part instanceof vscode.LanguageModelToolCallPart) {
        const id = part.callId || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        let args = '{}';
        try {
          args = JSON.stringify(part.input ?? {});
        } catch {
          args = '{}';
        }
        toolCalls.push({ id, type: 'function', function: { name: part.name, arguments: args } });
      } else if (isToolResultPart(part)) {
        const callId = (part as { callId?: string }).callId ?? '';
        const content = collectToolResultText(part as { content?: ReadonlyArray<unknown> });
        toolResults.push({ callId, content });
      }
    }

    let emittedAssistantToolCall = false;
    if (toolCalls.length > 0) {
      out.push({ role: 'assistant', content: textParts.join('') || undefined, tool_calls: toolCalls });
      emittedAssistantToolCall = true;
    }

    for (const tr of toolResults) {
      out.push({ role: 'tool', tool_call_id: tr.callId, content: tr.content || '' });
    }

    const text = textParts.join('');
    if (text && (role === 'system' || role === 'user' || (role === 'assistant' && !emittedAssistantToolCall))) {
      out.push({ role, content: text });
    }
  }
  return out;
}

/**
 * Convert VS Code tool definitions to OpenAI function tool definitions.
 * @param options Request options containing tools and toolMode.
 */
export function convertTools(options: vscode.LanguageModelChatRequestHandleOptions): { tools?: OpenAIFunctionToolDef[]; tool_choice?: 'auto' | { type: 'function'; function: { name: string } } } {
  const tools = options.tools ?? [];
  if (!tools || tools.length === 0) {
    return {};
  }

  const toolDefs: OpenAIFunctionToolDef[] = tools.map(t => {
    const hasParams = !!(t.inputSchema && typeof t.inputSchema === 'object' && Object.keys(t.inputSchema as Record<string, unknown>).length);
    return {
      type: 'function' as const,
      function: {
        name: t.name,
        description: t.description,
        parameters: hasParams ? t.inputSchema : undefined,
      },
    };
  });

  let tool_choice: 'auto' | { type: 'function'; function: { name: string } } = 'auto';
  if (options.toolMode === vscode.LanguageModelChatToolMode.Required) {
    if (tools.length !== 1) {
      console.error('ToolMode.Required but multiple tools:', tools.length);
      throw new Error('LanguageModelChatToolMode.Required is not supported with more than one tool');
    }
    tool_choice = { type: 'function', function: { name: tools[0].name } };
  }

  return { tools: toolDefs, tool_choice };
}

/**
 * Validate tool names to ensure they contain only word chars, hyphens, or underscores.
 * @param tools Tools to validate.
 */
export function validateTools(tools: readonly vscode.LanguageModelChatTool[]): void {
  for (const tool of tools) {
    if (!tool.name.match(/^[\w-]+$/)) {
      console.error(' Invalid tool name detected:', tool.name);
      throw new Error(`Invalid tool name "${tool.name}": only alphanumeric characters, hyphens, and underscores are allowed.`);
    }
  }
}

/**
 * Validate the request message sequence for correct tool call/result pairing.
 * @param messages The full request message list.
 */
export function validateRequest(messages: readonly vscode.LanguageModelChatRequestMessage[]): void {
  const lastMessage = messages[messages.length - 1];
  if (!lastMessage) {
    console.error('No messages in request');
    throw new Error('Invalid request: no messages.');
  }

  messages.forEach((message, i) => {
    if (message.role === vscode.LanguageModelChatMessageRole.Assistant) {
      const toolCallIds = new Set(
        message.content
          .filter(part => part instanceof vscode.LanguageModelToolCallPart)
          .map(part => (part as unknown as vscode.LanguageModelToolCallPart).callId)
      );
      if (toolCallIds.size === 0) {
        return;
      }

      let nextMessageIdx = i + 1;
      const errMsg = 'Invalid request: Tool call part must be followed by a User message with a LanguageModelToolResultPart with a matching callId.';
      while (toolCallIds.size > 0) {
        const nextMessage = messages[nextMessageIdx++];
        if (!nextMessage || nextMessage.role !== vscode.LanguageModelChatMessageRole.User) {
          console.error('Validation failed: missing tool result for call IDs:', Array.from(toolCallIds));
          throw new Error(errMsg);
        }

        nextMessage.content.forEach(part => {
          if (!isToolResultPart(part)) {
            const ctorName = (Object.getPrototypeOf(part as object) as { constructor?: { name?: string } } | undefined)?.constructor?.name ?? typeof part;
            console.error('Validation failed: expected tool result part, got:', ctorName);
            throw new Error(errMsg);
          }
          const callId = (part as { callId: string }).callId;
          toolCallIds.delete(callId);
        });
      }
    }
  });
}

/**
 * Type guard for LanguageModelToolResultPart-like values.
 * @param value Unknown value to test.
 */
export function isToolResultPart(value: unknown): value is { callId: string; content?: ReadonlyArray<unknown> } {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const obj = value as Record<string, unknown>;
  const hasCallId = typeof obj.callId === 'string';
  const hasContent = 'content' in obj;
  return hasCallId && hasContent;
}

/**
 * Map VS Code message role to OpenAI message role string.
 * @param message The message whose role is mapped.
 */
function mapRole(message: vscode.LanguageModelChatRequestMessage): Exclude<OpenAIChatRole, 'tool'> {
  const USER = vscode.LanguageModelChatMessageRole.User as unknown as number;
  const ASSISTANT = vscode.LanguageModelChatMessageRole.Assistant as unknown as number;
  const r = message.role as unknown as number;
  if (r === USER) { return 'user'; }
  if (r === ASSISTANT) { return 'assistant'; }
  return 'system';
}

/**
 * Concatenate tool result content into a single text string.
 * @param pr Tool result-like object with content array.
 */
function collectToolResultText(pr: { content?: ReadonlyArray<unknown> }): string {
  let text = '';
  for (const c of (pr.content ?? [])) {
    if (c instanceof vscode.LanguageModelTextPart) {
      text += c.value;
    } else if (typeof c === 'string') {
      text += c;
    } else {
      try { text += JSON.stringify(c); } catch { /* ignore */ }
    }
  }
  return text;
}

/**
 * Try to parse a JSON object from a string.
 * @param text The input string.
 * @returns Parsed object or ok:false.
 */
export function tryParseJSONObject(text: string): { ok: true; value: Record<string, unknown> } | { ok: false } {
  try {
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
