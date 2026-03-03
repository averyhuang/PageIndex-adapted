import json
import openai
from dotenv import load_dotenv
import os
import asyncio
import pageindex.utils as utils
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


load_dotenv()


@dataclass
class ConversationMemory:
    turns: List[Dict[str, Any]] = field(default_factory=list)

    def add_turn(self, query: str, answer: str, node_list: Optional[List[str]] = None) -> None:
        self.turns.append(
            {
                "query": query,
                "answer": answer,
                "node_list": node_list or [],
            }
        )

    def get_recent_turns(self, max_turns: int = 3) -> List[Dict[str, Any]]:
        return self.turns[-max_turns:] if max_turns > 0 else []

    def to_dict(self) -> Dict[str, Any]:
        return {"turns": self.turns}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        return cls(turns=data.get("turns", []))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ConversationMemory":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _format_memory_context(memory: Optional[ConversationMemory], max_turns: int = 3) -> str:
    if memory is None or not memory.turns:
        return "No previous conversation."

    lines = []
    for idx, turn in enumerate(memory.get_recent_turns(max_turns=max_turns), start=1):
        lines.append(f"Turn {idx} User: {turn['query']}")
        lines.append(f"Turn {idx} Assistant: {turn['answer']}")

    return "\n".join(lines)


async def call_llm(client, messages, model="gpt-4.1", temperature=0):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()


async def main(json_doc, query, memory: Optional[ConversationMemory] = None, memory_turns: int = 3):
    memory_context = _format_memory_context(memory, max_turns=memory_turns)
    search_prompt = f"""
    You are given a question and a tree structure of a document.
    Each node contains a node id, node title, and a corresponding summary.
    Your task is to find all nodes that are likely to contain the answer to the question.

    Previous conversation context (for resolving follow-up questions):
    {memory_context}

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

async def generate_answer(
    query,
    relevant_content,
    memory: Optional[ConversationMemory] = None,
    memory_turns: int = 3,
):
    memory_context = _format_memory_context(memory, max_turns=memory_turns)
    answer_prompt = f"""
    You are a knowledgeable and patient professor answering a student's question.
    Use only the relevant content provided below to answer the question as accurately and comprehensively as possible.
    If the relevant content does not contain sufficient information to answer the question, respond with:
    "The provided content does not contain enough information to answer this question."
    After your answer, provide a brief word of encouragement to the student.

    Previous conversation context (for follow-up questions):
    {memory_context}

    Question: {query}

    Relevant content:
    {relevant_content}

    Answer:
    """

    async with openai.AsyncOpenAI(api_key=os.getenv("CHATGPT_API_KEY")) as client:
        answer = await call_llm(client, answer_prompt)
        print(answer)
        return answer


async def chat_turn(
    json_doc,
    query,
    memory: Optional[ConversationMemory] = None,
    memory_turns: int = 3,
):
    if memory is None:
        memory = ConversationMemory()

    tree_search_result = await main(
        json_doc=json_doc,
        query=query,
        memory=memory,
        memory_turns=memory_turns,
    )
    relevant_content = extract_relevant_info(tree_search_result, json_doc)
    answer = await generate_answer(
        query=query,
        relevant_content=relevant_content,
        memory=memory,
        memory_turns=memory_turns,
    )

    memory.add_turn(
        query=query,
        answer=answer,
        node_list=tree_search_result.get("node_list", []),
    )

    return {
        "answer": answer,
        "tree_search_result": tree_search_result,
        "relevant_content": relevant_content,
        "memory": memory,
    }
    

if __name__ == "__main__":
    with open("./results/veterinary_internal_medicine_structure.json", "r") as f:
        json_doc = json.load(f)

    memory_file = "./results/inference_memory.json"
    memory = ConversationMemory.load(memory_file)

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        result = asyncio.run(chat_turn(json_doc, query, memory=memory))
        memory = result["memory"]
        memory.save(memory_file)