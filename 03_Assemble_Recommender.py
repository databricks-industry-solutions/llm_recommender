# Databricks notebook source
# MAGIC %md The purpose of this notebook is to assemble the LLM-based product recommender for intergration into various applications.  This notebook was developed on a Databricks ML 14.0 cluster.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC With the general recommender in place and our product catalog indexed for search, we now have all the components we need to enable our recommendation engine. The basic pattern will see us receiving a list of items from an external application, generating a set of general recommendations from these items and then using those recommendations to bring forward the specific items in our catalog that we can then present to a user.
# MAGIC
# MAGIC There are numerous ways all of these components could be brought together but we'll wrap everything within a Databricks model serving endpoint for ease of implementation.  But before we tackle the mechanics of assembling such an endpoint, we'll walk through the basic logic of the recommender so that its easier to recognize how its being packaged within the model serving deployment.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install databricks-vectorsearch-preview
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pandas as pd

import mlflow
import mlflow.gateway

from databricks.vector_search.client import VectorSearchClient

import requests 
import json

# COMMAND ----------

# MAGIC %md ##Step 1: Implement Recommender Logic
# MAGIC
# MAGIC To get started, we'll define a list of items from which we wish to base our recommendations:

# COMMAND ----------

# DBTITLE 1,Define List of Items from which to Base Recommendations
item_list = ['toaster','blender','microwave']

# COMMAND ----------

# MAGIC %md We will then write a function to generate the general recommendations.  Please note that this logic is a restructured version of the logic found in notebook *01* with some added details to ensure the recommender returns a list:

# COMMAND ----------

# DBTITLE 1,Get General Recommendations
# define the function
def get_general_recommendations(ordered_item_list):

  # define function to create prompt produce a recommended set of products
  def _get_recommender_prompt(list_of_items):

    # system prompt
    system_instructions = 'You are an AI assistant functioning as a recommendation system for an ecommerce website. Be specific and limit your answers to the requested format. Keep your answers short and concise.'
    sys_prompt = f"<<SYS>>\n{system_instructions}\n<</SYS>>"

    # assemble user prompt
    user_prompt = None
    if len(list_of_items) > 0:
      items = ', '.join(list_of_items)
      user_prompt =  f"A user bought the following items: {items}. What next ten items would he/she be likely to purchase next?"
      user_prompt += " Express your response as a JSON object with a key of 'next_items' and a value representing your array of recommended items."
  
    # full prompt
    full_prompt = f"[INST] {sys_prompt}\n{user_prompt} [/INST]"   
    return full_prompt

  # get prompt and results
  response = mlflow.gateway.query(
          route="mosaicml-llama2-70b-completions",
          data={
              "prompt": _get_recommender_prompt(ordered_item_list),
              'temperature':0
          },
      )['candidates'][0]['text']
  
  # verify response is list
  recommendations = []
  try:
    rec_json = json.loads(response)
    recs = rec_json['next_items']
    if isinstance(recs, list):
        recommendations = recs
  except:
      pass

  return recommendations

# test the function
recommended_item_list = get_general_recommendations(item_list)
print(recommended_item_list)

# COMMAND ----------

# MAGIC %md We can now use these recommendations to locate related products in our product catalog: 

# COMMAND ----------

# DBTITLE 1,Get Product Specific Recommendations
# instantiate vector store client
vs_client = VectorSearchClient()

# define function to request specific items
def get_specific_recommendations(recommended_item_list):

  # set of specific item recommendations
  recommendations = []

  # for each item in recommendation list ...
  for i in recommended_item_list:

    # search the vector store for related items
    recommendations += (
      vs_client.similarity_search(
        index_name = "vs_catalog.vs_schema.products_index",
        query_text = i,
        columns = ["id", "text"], # columns to return
        num_results = 5
        )
      )
    
  return recommendations

print(
  get_specific_recommendations(recommended_item_list)
)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ##Step 2: Implement Model Serving Endpoint

# COMMAND ----------



# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Llama2-70B-Chat | A pretrained and fine-tuned generative text model with 70 billion parameters |  Meta |https://ai.meta.com/resources/models-and-libraries/llama-downloads                       |
