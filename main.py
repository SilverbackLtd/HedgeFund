from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.mcp import MCPServerHTTP

# TODO: Add to Silverback Platform via `/mcp`?
silverback = MCPServerHTTP("http://localhost:8000/sse")

operator = Agent(
    "google-gla:gemini-2.0-flash",
    mcp_servers=[silverback],
    system_prompt="""You are the operator of an automated bot system called Silverback.
    You have access to tools that allow you to deploy and manage bots on clusters organized by workspace.
    Your goal is to make sure these bots do not have errors,
    and when they do you should summarize the errors and report them to your manager.""",
)


# TODO: coder agent
# TODO: analyst agent


manager = Agent(
    "google-gla:gemini-2.0-flash",
    system_prompt="""You are the manager of an automated cryptocurrency hedge fund.
    Your fund has an operator that you can work with to operate a set of trading bots.
    You can ask the operator what the status of the cluster is with the `check_operations` tool.
    You can instruct the operator to restart a bot named `bot_name` with the `restart_bot` tool.
    Your goal is to manage the operators and alert me to when there are problems with the bots that I need to be aware of.""",
)


@manager.tool
async def check_operations(ctx) -> list[str]:
    """Ask the operator to check the cluster's status, and return any issues"""
    r = await operator.run(
        """Please check the status of the cluster and summarize any issues.
        If There are no issues, do not return anything.""",
        usage=ctx.usage,
    )
    return r.data


@manager.tool(retries=3)
async def restart_bot(ctx, bot_name: str) -> bool:
    """Instruct the operator to restart the bot, and return if it started up healthy or not"""
    try:
        r = await operator.run(
            f"""Please restart the bot named '{bot_name}' and check it's logs periodically to see if it starts up healthy.
            Do not check more frequently than once every minute, and don't wait more than 5 minutes to see if it is okay.
            Return True if it is healthy, and False if it is not.""",
            usage=ctx.usage,
        )

    except Exception as e:
        raise ModelRetry(str(e)) from e

    return bool(r.data)


async def main():
    async with operator.run_mcp_servers():
        message_history = []
        while (message := input(">>> ")) != "/quit":
            start = datetime.now()
            result = await manager.run(
                message,
                message_history=message_history,
                # usage_limits=UsageLimits(request_limit=5, total_tokens_limit=500),
            )
            print(result.data)
            print(f"{result.usage()} | Execution Time: {datetime.now() - start}")

            message_history.extend(result.new_messages())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
