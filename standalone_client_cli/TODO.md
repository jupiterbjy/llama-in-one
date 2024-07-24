1. Implement chat summarize feature before context limit
   - extract system messages, do not summarize those.
   - move all system messages to start of the chat history.
   - summarize 1/4 of the message, using one Prompt/Reply cycle as unit.
   - inform user in some way, since this will invalidate token cache after summary so it takes bit more time.
2. meow bark woof