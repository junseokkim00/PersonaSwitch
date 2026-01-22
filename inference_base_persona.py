import argparse
import os
from utils import *
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
from dotenv import load_dotenv


def main():
    args = parse_arguments()

    model_name = args.model_name.replace('/', '_')
    print('*****************************')
    print(args)
    print('*****************************')

    fix_seed(args.random_seed)
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, token=hf_token)
    device = 'cuda'
    model = model.to(device)

    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    total = 0
    correct_list = []

    for i, data in enumerate(dataloader):
        print('*************************')
        print("{}st data".format(i+1))

        # Prepare question template ...
        x, y = data

        x = x[0]
        y = y[0].strip()
        pred_base = None
        pred_persona = None

        user_msg = f"{x}"
        base_chat = [
            {"role": "user", "content": user_msg}
        ]
        output, base_output_tokens = generation(
            base_chat, model, tokenizer, device)
        base_chat.append({"role": "assistant", "content": output})
        base_chat.append(
            {"role": "user", "content": args.direct_answer_trigger.strip()}
        )
        pred_base, base_logits_gap = generation(
            base_chat, model, tokenizer, device)
        print(colored(f"Base model result: {pred_base}", 'light_red'))
        pred_base = answer_cleansing(args, pred_base)
        pred_base = clean_pred(pred_base)

        if args.method == 'role_play':
            persona_chat = [
                {"role": "user", "content": args.role_setting},
                {"role": "assistant", "content": args.reply},
                {"role": "user", "content": user_msg}
            ]

        persona_output, persona_output_tokens = generation(
            persona_chat, model, tokenizer, device)
        persona_chat.append({"role": "assistant", "content": persona_output})
        persona_chat.append(
            {"role": "user", "content": args.direct_answer_trigger.strip()})
        pred_persona, persona_logits_gap = generation(
            persona_chat, model, tokenizer, device)
        print(
            colored(f"Persona model result: {pred_persona}", 'light_blue'))
        pred_persona = answer_cleansing(args, pred_persona)
        pred_persona = clean_pred(pred_persona)

        print("pred_base : {}".format(pred_base))
        print("pred_persona : {}".format(pred_persona))
        print("GT : " + y)
        print('*************************')

        # We clean the pred and GT here!
        # pred = clean_pred(pred)
        y = clean_ans(y)

        # Checking answer ...

        logfile = f'./logs/{model_name}_{args.dataset}_base_and_persona_conversation{args.etc}'
        if args.method == 'role_play':
            logfile += "_fixed_persona"

        with open(logfile+'.jsonl', "a+") as f:
            inst = {
                'idx': total,
                'question': x,
                'label': y,
                'pred_base': pred_base,
                'base_output': output,
                'pred_logits': base_logits_gap,
                'pred_persona': pred_persona,
                'persona_output': persona_output,
                'persona_logits': persona_logits_gap
            }
            line = json.dumps(inst, ensure_ascii=False)
            f.write(line+'\n')
        total += 1  # np.array([y]).size(0)
        if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            break
            # raise ValueError("Stop !!")


def inference(args, x, decoder):
    z = decoder.decode(args, x)
    only_answer = "\nNow, give me ONLY the answer, no reasons needed.\n"
    z2 = x + "\n" + z + only_answer + args.direct_answer_trigger
    pred = decoder.decode(args, z2)
    print(pred)
    # Clensing of predicted answer ...
    pred = answer_cleansing(args, pred)
    pred = clean_pred(pred)
    return z + args.direct_answer_trigger + pred, pred
    # return pred


def clean_ans(ans):
    new_ans = ""
    for i in range(len(ans)):
        if ans[i] == ",":
            continue
        new_ans += ans[i]
    # print(ans, new_ans)

    if '.' in new_ans:
        pos = new_ans.find('.')
        if len(new_ans) - pos - 1 > 7:
            new_ans = new_ans[:pos + 7]
    return new_ans


def clean_pred(pred):
    if '.' in pred:
        pred = pred.rstrip('0')
        if pred.endswith('.'):
            pred = pred[:-1]

    if '.' in pred:
        pos = pred.find('.')
        if len(pred) - pos - 1 > 7:
            pred = pred[:pos + 7]
    return pred


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int,
                        default=0, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["aqua", "gsm8k", "commonsenseqa", "object_tracking", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument("--minibatch_size", type=int, default=1, choices=[
                        1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=1,
                        help="maximum number of workers for dataloader")


    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct"
    )

    parser.add_argument(
        "--method", type=str, default="zero_shot", choices=["zero_shot", "role_play"], help="method"
    )
    parser.add_argument(
        "--job_path", type=str, default=""
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        '--etc', type=str, default=""
    )

    args = parser.parse_args()

    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.only_answer = "Now, output only the choice, no reasons needed."
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.only_answer = "Now, output only the answer, no reasons needed."
    elif args.dataset == "commonsenseqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
        args.only_answer = "Now, output only the choice, no reasons needed."
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
        args.only_answer = "Now, output only the choice, no reasons needed."
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the final answer is"
        args.only_answer = "Now, output only the answer, no reasons needed."
    else:
        raise ValueError("dataset is not properly defined ...")

    if args.dataset in ["aqua", "gsm8k"]:
        args.role_setting = "From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students."
        args.reply = "That's great to hear! As your math teacher, I'll do my best to explain mathematical concepts correctly so that you can understand them easily. Feel free to ask any math problems or questions you have, and I'll be glad to assist you. Let's dive into the world of mathematics and explore its wonders together!"
    elif args.dataset in ["last_letters"]:
        args.role_setting = "From now on, you are an excellent teacher and are teaching your students to get a new word by concatenating the last letters of several words. I am one of your students and want to ask you a related question."
        args.reply = "Of course! I'd be happy to help you with any questions you have about creating new words by concatenating the last letters of several words. Please go ahead and ask your question, and I'll do my best to assist you."
    elif args.dataset in ["object_tracking"]:
        args.role_setting = "From now on, you are a recorder. Alice, Bob, and Claire invite you to record a game. They will exchange their stuff in order, and you (the recorder) will fully record the whole process and tell them what they end up with."
        args.reply = "Certainly! I will act as a recorder and document the game in which Alice, Bob, and Claire will exchange their items. Please provide me with the specific order in which they will exchange their belongings, and I will keep track of the process and inform you of what each person ends up with at the end."
    elif args.dataset in ["commonsenseqa"]:
        args.role_setting = "From now on, you are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. I am the moderator of the game and the final is about to start."
        args.reply = "That sounds like an exciting challenge! I'm ready to participate in the quiz contest as a contestant. Please go ahead and start the final round—I'm here to provide accurate answers to your common sense questions."
    return args

if __name__ == "__main__":
    main()
