import json
import os

from dotenv import load_dotenv

from src.pipeline import build_ma_rag_graph, build_retriever_tool, dump_json, normalize_question
from src.utils import load_benchmark_dataset, parse_args

load_dotenv()


if __name__ == "__main__":
    args = parse_args()
    retriever_tool = build_retriever_tool(gpu_ids=args.gpus)
    graph = build_ma_rag_graph(retriever_tool)

    dataset_name = args.dataset
    dataset = load_benchmark_dataset(dataset_name)

    save_dir = f"{args.exp}_{args.model}_{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    for id, item in enumerate(dataset):
        if id < args.start_index or id > args.end_index:
            continue
        question_id = item["id"]
        print(question_id)
        save_file = os.path.join(save_dir, f"{question_id}.json")
        if os.path.exists(save_file):
            continue
        if dataset_name == "fever":
            question = f"Verify this claim, answer SUPPORTS or REFUTES\n{item['input']}"
        else:
            question = item["input"]
        inputs = {"original_question": normalize_question(question)}
        try:
            output = graph.invoke(inputs)
            print(output)
            print()
            dump_json(save_file, output)
        except Exception as e:
            print(e)
