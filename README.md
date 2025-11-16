# Daily Standup Bot

AI-powered daily standup reports from Git commits. This VS Code extension automatically generates professional standup summaries from your git commits using OpenAI or Google Gemini, and can send them directly to Slack.

## Features

- 🤖 **AI-Powered Summaries**: Uses OpenAI GPT or Google Gemini to generate intelligent commit summaries
- 📅 **Daily Standup Reports**: Automatically analyzes today's commits and creates standup-ready summaries
- 🔄 **Multiple AI Providers**: Choose between OpenAI (ChatGPT) or Google Gemini
- 📱 **Slack Integration**: Send summaries directly to Slack channels via webhook or bot token
- 🔐 **Secure Storage**: API keys stored securely using VS Code's encrypted storage
- ⚙️ **Easy Configuration**: Simple setup through VS Code commands and settings

## How It Works

1. **Configure AI Provider**: Set up your OpenAI or Gemini API key
2. **Optional Slack Setup**: Configure Slack webhook or bot token for team sharing
3. **Generate Standup**: Run the command to analyze today's commits
4. **Share Results**: View, copy, or send the AI-generated summary to Slack

## Commands

Access these commands via the Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`):

- `Daily Standup Bot: Generate Daily Standup` - Create AI summary from today's commits
- `Daily Standup Bot: Configure OpenAI API Key` - Set up OpenAI integration
- `Daily Standup Bot: Configure Gemini API Key` - Set up Google Gemini integration
- `Daily Standup Bot: Switch AI Provider` - Toggle between OpenAI and Gemini
- `Daily Standup Bot: Configure Slack Integration` - Set up Slack webhook or bot token
- `Daily Standup Bot: Test Slack Connection` - Verify Slack configuration

## Requirements

- VS Code 1.101.0 or higher
- Git repository with commit history
- API key for either:
  - OpenAI (ChatGPT) - Get from [OpenAI Platform](https://platform.openai.com/api-keys)
  - Google Gemini - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Extension Settings

This extension contributes the following settings:

- `daily-standup-bot.provider`: Choose between "openai" or "gemini"
- `daily-standup-bot.openai.model`: OpenAI model selection (gpt-3.5-turbo, gpt-4, etc.)
- `daily-standup-bot.gemini.model`: Gemini model selection (gemini-pro, gemini-2.0-flash, etc.)
- `daily-standup-bot.slack.defaultChannel`: Default Slack channel for messages
- `daily-standup-bot.slack.botUsername`: Display name for the bot in Slack
- `daily-standup-bot.slack.botIcon`: Emoji icon for the bot in Slack

## Setup Guide

### 1. Install the Extension

Install "Daily Standup Bot" from the VS Code Extension Marketplace.

## Local Development

### Prerequisites
- VS Code 1.101.0+
- Node.js 18+ and npm

### Install dependencies
```bash
npm install
```

### Start in development (watch mode)
```bash
npm run watch
```

### Run the extension (Extension Development Host)
- Press F5 in VS Code, or use Run and Debug → “Run Extension”.

Alternatively, via CLI:
```bash
code --extensionDevelopmentPath="PATH/TO/daily-standup-bot"
```

### Lint
```bash
npm run lint
```

### Test
```bash
npm test
```

### Build (production bundle)
```bash
npm run build
```

### 2. Configure AI Provider

Choose your preferred AI service:

**For OpenAI:**

1. Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Run `Daily Standup Bot: Configure OpenAI API Key`
3. Select your preferred model

**For Google Gemini:**

1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Run `Daily Standup Bot: Configure Gemini API Key`
3. Select your preferred model

### 3. Optional: Configure Slack

To send summaries to Slack:

**Webhook Method (Recommended):**

1. Create a Slack webhook in your workspace
2. Run `Daily Standup Bot: Configure Slack Integration`
3. Choose "Webhook URL" and paste your webhook

**Bot Token Method:**

1. Create a Slack app with bot permissions
2. Run `Daily Standup Bot: Configure Slack Integration`
3. Choose "Bot Token" and paste your bot token

### 4. Generate Your First Standup

1. Make sure you have commits from today
2. Run `Daily Standup Bot: Generate Daily Standup`
3. Choose how to share the results

## Privacy & Security

- All API keys are stored securely using VS Code's encrypted storage
- No commit data is stored permanently
- All communication with AI services is encrypted
- You maintain full control over your data

## Release Notes

### 1.0.0

Initial release featuring:

- AI-powered commit summarization
- OpenAI and Google Gemini integration
- Slack webhook and bot token support
- Secure credential storage
- Multiple output formats (view, copy, send to Slack)

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**Enjoy your automated daily standups!** 🚀
