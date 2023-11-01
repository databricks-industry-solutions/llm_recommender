# Databricks notebook source
# MAGIC %md The purpose of this notebook is to connect to an LLM in order to generate generalized product recommendations.  This notebook was developed on a Databricks ML 14.0 cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC The first step in building our recommendations application is to connect to a large language model (LLM) and construct a prompt capable of producing a list of product recommendations.  In this notebook, we will leverage the MLFlow AI Gateway to construct a managed connection to an LLM hosted as part of a third-party service.  We could train and/or host our own LLM for this purpose, but there appears to be little advantage in doing so, especially given the cost of performing that work to achieve a generalized knowledge.
# MAGIC
# MAGIC Once we have established a connection to a hosted LLM, we will explore the structure of a prompt capable of taking a set of items and from them returning a list of suggestions.

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# DBTITLE 1,Import the Required Libraries
import mlflow.gateway

# COMMAND ----------

# MAGIC %md ##Step 1: Setup Route to LLM
# MAGIC
# MAGIC With the availability of a wide variety of proprietary services and open source models, we have numerous options for how we will address our LLM needs. The MLFlow AI Gateway provides us a mechanism for abstracting connectivity to a model and simplifying how we make calls to it.  Down the road, should we decide to employ a different model, this abstraction will allow us to make a substitution with minimum code changes.  The gateway also provides security detail obfuscation and rate limiting capabilities which make our deployment more secure.
# MAGIC
# MAGIC For this solution accelerator, we've decided to make use of [Meta's Llama2-70B-Chat model](https://ai.meta.com/llama/), a leading open source model comparable to the proprietary OpenAI ChatGPT and Google PaLM-Bison. The size of this model makes use of a hosted service desirable so that we'll make use of a version of this model [hosted on MosaicML's inference layer](https://www.mosaicml.com/blog/llama2-inference).  Our first step is to **request access to this service** by completing [this form](https://www.mosaicml.com/get-started).
# MAGIC
# MAGIC Once we have access to this service, we need to **secure the API key** for this endpoint following the instructions [provided here](https://docs.mosaicml.com/en/latest/inference.html). This key should be secured using the Databricks secrets capability as [documented here](https://docs.databricks.com/en/security/secrets/index.html). In the steps below, the API key will be secured with a scope of *llm_recommender* and a key of *mosaicml_api_key*.  Assuming you have the Databricks CLI installed and configured to connect to your workspace, you can setup this secret with these commands:
# MAGIC ```
# MAGIC databricks secrets create-scope llm_recommender
# MAGIC databricks secrets put-secret llm_recommender mosaicml_api_key
# MAGIC ```

# COMMAND ----------

# MAGIC %md With the key securely in place, we can then proceed with the setting up of an MLFlow AI Gateway route.  [MLFlow](https://mlflow.org/) is an open source model management and deployment platform.  Databricks comes pre-configured with a managed instance of MLFlow, making it very simple to connect to the AI Gateway feature within it:

# COMMAND ----------

# DBTITLE 1,Identify MLFlow AI Gateway
# set mlflow ai to this databricks workspace
mlflow.gateway.set_gateway_uri("databricks")

# COMMAND ----------

# MAGIC %md We now setup a route within the [AI Gateway](https://mlflow.org/docs/latest/gateway/index.html) to point to the MosaicML service. A key consideration in setting up the gateway is the *route_type*.  As of the time of notebook development, the available *route_type* options were:
# MAGIC </p>
# MAGIC
# MAGIC * *llm/v1/completions* - respond to an isolated prompt
# MAGIC * *llm/v1/chat* - respond to a prompt that's part of an ongoing conversation
# MAGIC * *llm/v1/embeddings* - respond with the embedding vector associated with a unit of text
# MAGIC
# MAGIC The choice of *route_type* (along with the functions supported by the model) dictates the structure of the prompt supplied to the model as well as the output returned.  The *completions* and *chat* prompts return text in response to a prompt while the *embeddings* type returns the vector representing the embedding associated with a supplied unit of text.  Between *completions* and *chat*, the choice of route type is determined by whether we want a response to a single user input or if we wish the response to include knowledge of a sequence of exchanges preceding the latest prompt.  As we will not be engaging in an on-going conversation with the model, our *route_type* will be the *completion* type.
# MAGIC
# MAGIC For the *model* definition, we need to provide a dictionary adhering to the forms and associated with supported providers as documented [here](https://mlflow.org/docs/latest/gateway/index.html#providers):

# COMMAND ----------

# DBTITLE 1,Setup Route to MosaicML Endpoint
try: 
     # define route to mosaicml endpoint
    mlflow.gateway.create_route(
        name="mosaicml-llama2-70b-completions", # the name we wish to refer to this route by
        route_type="llm/v1/completions", # the route type appropriate for this engagement
        model={
            "name": "llama2-70b-chat", 
            "provider": "mosaicml",
            "mosaicml_config": {
                "mosaicml_api_key": dbutils.secrets.get(scope="llm_recommender", key="mosaicml_api_key"),
            },
        },
    )
except: # if route already exists, you'll see an error message but will land here
    pass

# COMMAND ----------

# MAGIC %md With the route defined, we can provide it a prompt, adhering to the prompt standards associated with the Llama2 model to solicit a response:
# MAGIC
# MAGIC **NOTE** We will discuss the prompt structure later in this notebook.

# COMMAND ----------

# DBTITLE 1,Test the Route
# this is a test prompt in the format required by Llama2
prompt = """[INST] <<SYS>>
You are an AI assistant functioning as a recommendation system for an ecommerce website. Keep your answers short and concise.
<</SYS>>
A user bought a pen, pencil, and a glue stick in that order. What item would he/she be likely to purchase next?[/INST]
"""

# query the completions route using the mlflow client
print(
    mlflow.gateway.query(
        route="mosaicml-llama2-70b-completions",
        data={
            "prompt": prompt,
            'temperature':0
        },
    )
)

# COMMAND ----------

# MAGIC %md ##Step 2:  Engineer the prompt
# MAGIC
# MAGIC With our route established, we can now start working on a prompt.  The basic structure of a [prompt as expected by Llama2](https://huggingface.co/blog/llama2#how-to-prompt-llama-2) is as follows. Please note that the line-breaks are expected in the prompt string:
# MAGIC ```
# MAGIC [INST] <<SYS>>
# MAGIC {{ system_prompt }}
# MAGIC <</SYS>>
# MAGIC
# MAGIC {{ user_message }} [/INST]
# MAGIC ```
# MAGIC
# MAGIC The *INST* tags identify the text as an instruction.  The instruction is divided into system and user portions. 
# MAGIC The system prompt, wrapped within the *SYS* markers provides general guidance for the model, indicating its high-level behavior.  The user prompt that follows provides the specific prompt it should respond to.
# MAGIC
# MAGIC This is probably the most challenging part of this exercise in that we want to trigger the LLM to return meaningful responses AND do so in a format that's easy for us to work with.  As you may see in the LLM output generated as part of the test at the end of the last step, the LLM has a preference to generate responses using natural language structures.
# MAGIC
# MAGIC To construct a high-quality prompt, you should experiment with a number of variations, making small adjustments to the instructions to see what gives you the best results in terms of both item suggestions and output formats.  Please note, that misspellings and other errors can have unintended consequences for your results so take care that your prompt reflects your intentions as you adjust the language used.
# MAGIC
# MAGIC For our recommender's needs, we should explore how to construct a prompt where we provide a list of items and then ask the LLM to tell us what likely *comes next*. 

# COMMAND ----------

# DBTITLE 1,Define Function to Build Prompt
# define function to create prompt produce a recommended set of products
def get_recommender_prompt(ordered_list_of_items):

  # system prompt
  system_instructions = 'You are an AI assistant functioning as a recommendation system for an ecommerce website. Be specific and limit your answers to the requested format. Keep your answers short and concise.'
  sys_prompt = f"<<SYS>>\n{system_instructions}\n<</SYS>>"

  # assemble user prompt
  user_prompt = None
  if len(ordered_list_of_items) > 0:
    items = ', '.join(ordered_list_of_items)
    user_prompt =  f"A user bought the following items: {items}. What next ten items would he/she be likely to purchase next?"
    user_prompt += " Express your response as a JSON object with a key of 'next_items' and a value representing your array of recommended items."
 
  # full prompt
  full_prompt = f"[INST] {sys_prompt}\n{user_prompt} [/INST]"   
  return full_prompt

# COMMAND ----------

# DBTITLE 1,Test the Prompt
# get prompt and results
prompt = get_recommender_prompt(
    ['scarf', 'beanie', 'ear muffs', 'thermal underwear']
    )
response = mlflow.gateway.query(
        route="mosaicml-llama2-70b-completions",
        data={
            "prompt": prompt,
            'temperature':0
        },
    )['candidates'][0]['text']

# print prompt and results
print(f"PROMPT:/n{prompt}", '\n')
print(
  f"RESPONSE:\n{response}" # extract the text from the model's response
)

# COMMAND ----------

# MAGIC %md Getting the model to respond with lists in the structure  you want and with relevant content is tricky.  (For example, we couldn't get a valid dictionary structure by requesting a python dictionary but found that requesting a JSON object did the trick.) You'll need to experiment with a variety of wordings and phrasings to trigger the right results.

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |
