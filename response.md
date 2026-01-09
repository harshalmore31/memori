Your approach is on the right track! A few adjustments: use `mem.llm.register(openai_chat=model)` instead of `mem.agno.register()` as the latter is deprecated. The main issue with your code is a potential race condition in async Discord bots - `mem.attribution()` sets shared config state, so concurrent messages from different users could overwrite each other's attribution before `agent.run()` executes.

The fix is to create isolated `Memori` and model instances per request:

```python
@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Create isolated instances per request
    model = OpenAIChat(id="gpt-4o-mini")
    mem = Memori(conn=Session).llm.register(openai_chat=model)

    mem.attribution(
        entity_id=f"discord_user_{message.author.id}",
        process_id=f"discord_channel_{message.channel.id}"
    )

    agent = Agent(model=model, instructions=["Be helpful"], markdown=True)
    response = agent.run(message.content)

    mem.config.augmentation.wait()  # Ensure fact extraction completes
    await message.channel.send(response.content)
```

Under the hood, `mem.llm.register()` wraps the underlying OpenAI client (not Agno itself). When `agent.run()` executes, Memori intercepts the LLM call to inject relevant facts from the user's memory and conversation history, then stores new messages and extracts facts asynchronously. Your `entity_id` per user and `process_id` per channel pattern is correct - this ensures each user has isolated memories while sharing the channel context.
