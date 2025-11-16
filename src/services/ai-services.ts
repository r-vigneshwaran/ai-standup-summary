import { generateText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createGoogleGenerativeAI } from "@ai-sdk/google";

interface AIResponse {
  content: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

interface AIServiceConfig {
  apiKey: string;
  model?: string;
}

export class OpenAIService {
  private apiKey: string;
  private model: string;
  private client: ReturnType<typeof createOpenAI>;

  constructor(config: AIServiceConfig) {
    this.apiKey = config.apiKey;
    this.model = config.model || "gpt-3.5-turbo";
    this.client = createOpenAI({ apiKey: this.apiKey });
  }

  async generateResponse(
    prompt: string,
    systemPrompt?: string
  ): Promise<AIResponse> {
    const mergedPrompt = systemPrompt ? `${systemPrompt}\n\n${prompt}` : prompt;

    try {
      const result = await generateText({
        model: this.client(this.model),
        prompt: mergedPrompt,
        temperature: 0.7,
        maxOutputTokens: 1000,
      });

      return {
        content: result.text,
        usage: {
          promptTokens: result.usage?.inputTokens ?? 0,
          completionTokens: result.usage?.outputTokens ?? 0,
          totalTokens:
            result.usage?.totalTokens ??
            (result.usage?.inputTokens ?? 0) +
              (result.usage?.outputTokens ?? 0),
        },
      };
    } catch (error) {
      console.error("OpenAI Service Error:", error);
      throw new Error(
        `Failed to generate response: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  async summarizeCommits(commits: string[]): Promise<string> {
    const prompt = `Please summarize the following git commits into a concise daily standup update:

${commits.join("\n")}

Format the summary as a brief, professional update suitable for a team standup meeting.`;

    const systemPrompt =
      "You are an expert at summarizing software development progress for daily standup meetings. Create concise, clear summaries that highlight key accomplishments and changes.";

    const response = await this.generateResponse(prompt, systemPrompt);
    return response.content;
  }
}

export class GeminiService {
  private apiKey: string;
  private model: string;
  private client: ReturnType<typeof createGoogleGenerativeAI>;

  constructor(config: AIServiceConfig) {
    if (!config.apiKey) {
      throw new Error("API key is required for GeminiService.");
    }
    this.apiKey = config.apiKey;
    this.model = config.model || "gemini-pro";
    this.client = createGoogleGenerativeAI({ apiKey: this.apiKey });
  }

  /**
   * Generates a response using the AI SDK Google provider.
   */
  async generateResponse(
    prompt: string,
    systemPrompt?: string
  ): Promise<AIResponse> {
    const mergedPrompt = systemPrompt ? `${systemPrompt}\n\n${prompt}` : prompt;

    try {
      const result = await generateText({
        model: this.client(this.model),
        prompt: mergedPrompt,
        temperature: 0.7,
        maxOutputTokens: 1000,
      });

      return {
        content: result.text,
        usage: {
          promptTokens: result.usage?.inputTokens ?? 0,
          completionTokens: result.usage?.outputTokens ?? 0,
          totalTokens:
            result.usage?.totalTokens ??
            (result.usage?.inputTokens ?? 0) +
              (result.usage?.outputTokens ?? 0),
        },
      };
    } catch (error) {
      console.error("Gemini Service Error:", error);
      throw new Error(
        `Failed to generate response: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  }

  /**
   * Creates a prompt to summarize commits and calls the generator.
   */
  async summarizeCommits(commits: string[]): Promise<string> {
    const prompt = `Please summarize the following git commits into a concise daily standup update:

${commits.join("\n")}

Format the summary as a brief, professional update suitable for a team standup meeting.`;

    const systemPrompt =
      "You are an expert at summarizing software development progress for daily standup meetings. Create concise, clear summaries that highlight key accomplishments and changes.";

    const response = await this.generateResponse(prompt, systemPrompt);
    return response.content;
  }
}

// Factory function to create AI service based on provider
export function createAIService(
  provider: "openai" | "gemini",
  config: AIServiceConfig
) {
  switch (provider) {
    case "openai":
      return new OpenAIService(config);
    case "gemini":
      return new GeminiService(config);
    default:
      throw new Error(`Unsupported AI provider: ${provider}`);
  }
}
