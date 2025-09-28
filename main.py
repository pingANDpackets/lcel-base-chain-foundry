# langchain_azure_example.py
from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Azure Foundry chat model
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_community.chat_models import ChatOllama



load_dotenv()


def main():
    """Summarize the provided biography with an Azure Foundry chat model."""
    print("Hello from langchain-course (Azure Foundry)!")
    # Sample biography we use to exercise the summarization chain.
    information = """
    Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman, known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion.

    Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada. He received bachelor's degrees from the University of Pennsylvania in 1997 before moving to California, United States, to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.

    In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI. In 2022, he acquired the social network Twitter, implementing significant changes and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017.

    Musk was the largest donor in the 2024 U.S. presidential election, and is a supporter of global far-right figures, causes, and political parties. In early 2025, he served as senior advisor to United States president Donald Trump and as the de facto head of DOGE. After a public feud with Trump, Musk left the Trump administration and announced he was creating his own political party, the America Party.

    Musk's political activities, views, and statements have made him a polarizing figure, especially following the COVID-19 pandemic. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation and promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE.
    """

    # Prompt template (same idea as your original)
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    # Bind the template variable so LangChain can inject the biography at runtime.
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # ---------- Azure Foundry / Azure OpenAI client ----------
    # Required environment variables:
    #   AZURE_INFERENCE_ENDPOINT  -> e.g. "https://<your-resource>.services.ai.azure.com/models"
    #   AZURE_INFERENCE_CREDENTIAL -> the key string (or you can use Entra auth; see below)
    #
    endpoint = os.environ.get("AZURE_INFERENCE_ENDPOINT")
    credential = os.environ.get("AZURE_INFERENCE_CREDENTIAL")

    if not endpoint or not credential:
        raise RuntimeError(
            "Set AZURE_INFERENCE_ENDPOINT and AZURE_INFERENCE_CREDENTIAL in your environment (or .env)."
        )

    # Choose the model name you deployed in Foundry (replace with your deployment)
    # If you deployed a model named "gpt-5" in Foundry, set model="gpt-5".
    model_name = os.environ.get("AZURE_MODEL_NAME", "gpt-5")

    llm = AzureAIChatCompletionsModel(
        endpoint=endpoint,
        credential=credential,
        model=model_name,
        # optional: temperature param may or may not be supported by the underlying model; try 0
        temperature=0.0,
    )

    # llm = ChatOllama(
    #     model="gemma:2b",
    #     base_url="http://localhost:11434",
    #     temperature=0.0
    # )

    # Compose LangChain prompt and model into a runnable pipeline.LCEL chain
    chain = summary_prompt_template | llm

    # Invoke the chain to request a summary from the Azure-hosted model.
    response = chain.invoke({"information": information})

    # response is a ChatModel result object. Print text content.
    # Different langchain versions may return different shapes; the object usually has .content or .text
    try:
        # prefer .content
        print(response.content)
    except Exception:
        # Fallback keeps compatibility with older LangChain return types that lack .content
        print(str(response))


if __name__ == "__main__":
    main()
