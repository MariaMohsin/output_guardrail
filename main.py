import asyncio
from pydantic import BaseModel
from agents import Agent,OpenAIChatCompletionsModel,OutputGuardrailTripwireTriggered,RunContextWrapper,Runner,output_guardrail,AsyncOpenAI, GuardrailFunctionOutput
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv(), override=True)
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


set_tracing_key=disabled = True
# Model initialization
model1 = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client)

class MessageOutput(BaseModel): 
    response: str

class MathOutput(BaseModel): 
    reasoning: str
    is_math: bool

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the output includes any math.",
    output_type=MathOutput,
    model=model1,
)

@output_guardrail
async def math_guardrail(  
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, output.response, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math,
    )

agent = Agent( 
    name="Customer Support Agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    model=model1,
    output_guardrails=[math_guardrail],
    output_type=MessageOutput,
)

async def main():
    # This should trip the guardrail
    try:
        result = await Runner.run(agent, "Hello, can you help me solve for x: 2x + 3 = 11?")
        print("Guardrail didn't trip - this is unexpected")

    except OutputGuardrailTripwireTriggered:
        print("Math output guardrail tripped")

if __name__ == "__main__":
    asyncio.run(main())
