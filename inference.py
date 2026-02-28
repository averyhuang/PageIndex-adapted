import json
import openai
from dotenv import load_dotenv
import os
import asyncio
import pageindex.utils as utils


load_dotenv()



async def call_llm(client, prompt, model="gpt-4.1", temperature=0):
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()


async def main(json_doc, query):
    search_prompt = f"""
    You are given a question and a tree structure of a document.
    Each node contains a node id, node title, and a corresponding summary.
    Your task is to find all nodes that are likely to contain the answer to the question.

    Question: {query}

    Document tree structure:
    {json.dumps(json_doc, indent=2)}

    Please reply in the following JSON format:
    {{
        "thinking": "<Your thinking process on which nodes are relevant to the question>",
        "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """

    async with openai.AsyncOpenAI(api_key=os.getenv("CHATGPT_API_KEY")) as client:
        tree_search_result = await call_llm(client, search_prompt)
        # Strip markdown code block wrappers if present
        tree_search_result = tree_search_result.strip()
        if tree_search_result.startswith("```"):
            tree_search_result = tree_search_result.split("\n", 1)[-1]
            tree_search_result = tree_search_result.rsplit("```", 1)[0].strip()
        # Parse the string into JSON
        tree_search_json = json.loads(tree_search_result)
        print(type(tree_search_json))
        print(tree_search_json)
        # Save to a json file
        # with open("./results/tree_search_result.json", "w") as f:
        #     json.dump(tree_search_json, f, indent=2)

    return tree_search_json


def extract_relevant_info(tree_search_result, json_doc):
    node_map = utils.create_node_mapping(json_doc)
    node_list = tree_search_result["node_list"]
    
    # Extract text from relevant nodes, ensuring node_id is 4 digits
    relevant_content = "\n\n".join(
        node_map[node_id]["summary"] 
        for node_id in node_list 
        if node_id in node_map
    )
    return relevant_content

async def generate_answer(query, relevant_content):
    answer_prompt = f"""
    You are a knowledgeable and patient professor answering a student's question.
    Use only the relevant content provided below to answer the question as accurately and comprehensively as possible.
    If the relevant content does not contain sufficient information to answer the question, respond with:
    "The provided content does not contain enough information to answer this question."
    After your answer, provide a brief word of encouragement to the student.

    Question: {query}

    Relevant content:
    {relevant_content}

    Answer:
    """

    async with openai.AsyncOpenAI(api_key=os.getenv("CHATGPT_API_KEY")) as client:
        answer = await call_llm(client, answer_prompt)
        print(answer)
        return answer
    

if __name__ == "__main__":
    with open("./results/veterinary_internal_medicine_structure.json", "r") as f:
        json_doc = json.load(f)
    
    # Step 1: Find relevant nodes using tree search
    query = "Which drugs require the most caution when interpreting thyroid panels?"
    tree_search_result = asyncio.run(main(json_doc, query))
    
    # Step 2: Extract relevant content from those nodes
    relevant_content = extract_relevant_info(tree_search_result, json_doc)
    
    # Step 3: Generate final answer
    final_answer = asyncio.run(generate_answer(query, relevant_content))