import argparse
import random
import os
from utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.corpus import stopwords
import nltk
import string
from dotenv import load_dotenv


def base_or_persona(args, question, response, model, tokenizer, device):
    user_msg = question

    chat = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": response}
    ]
    if args.method == "role_play":
        persona_chat = [
            {"role": "user", "content": args.role_setting},
            {"role": "assistant", "content": args.reply},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response}
        ]
    else:
        persona_chat = [
            {"role": "system", "content": f"You are a {args.job}."},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response}
        ]
    base_token, base_logits = generate_next_step(
        chat, model, tokenizer, device)
    persona_token, persona_logits = generate_next_step(
        persona_chat, model, tokenizer, device)
    return_token = None
    flag = None
    base_gap = 0
    persona_gap = 0
    if base_token == persona_token:
        response += base_token
        return_token = base_token
        flag = 0
    else:
        if args.compare == 'probs':
            base_gap = np.prod([logit[2] for logit in base_logits])
            persona_gap = np.prod([logit[2] for logit in persona_logits])
            if args.normalization:
                base_gap = np.log(base_gap) / len(base_logits)
                persona_gap = np.log(persona_gap) / len(persona_logits)
        else:
            base_gap = np.mean([logit[1] for logit in base_logits])
            persona_gap = np.mean([logit[1] for logit in persona_logits])
        if args.cmp == 'big':
            if base_gap > persona_gap:
                response += base_token
                return_token = base_token
                flag = 1
            else:
                response += persona_token
                return_token = persona_token
                flag = 2
        elif args.cmp == 'small':
            if base_gap < persona_gap:
                response += base_token
                return_token = base_token
                flag = 1
            else:
                response += persona_token
                return_token = persona_token
                flag = 2
        elif args.cmp == 'random':
            coin = random.choice([True, False])
            if coin:
                response += base_token
                return_token = base_token
                flag = 1
            else:
                response += persona_token
                return_token = persona_token
                flag = 2
    return response, return_token, flag, (base_gap, persona_gap)


def run_base_or_persona(args, question, model, tokenizer, device, max_token=512):
    response = ""
    count = [0, 0, 0]
    dnl_list = []
    cnt = 0
    while cnt <= max_token:
        response, last_token, flag, gap = base_or_persona(
            args, question, response, model, tokenizer, device)
        with open(f"./current_{args.dataset}_dnl.txt", "w") as f:
            f.write(
                f"Question: {question}\n\nresponse: {response}\n\nlast_token: {last_token}\n\ncnt: {cnt} / 512")
        count[flag] += 1
        dnl_list.append((last_token, flag, gap))
        if last_token == '':
            break
        cnt = len(tokenizer.encode(response)) - 1
    return response, count, dnl_list


# filter version
def filter_stopword(logits, stopword_ls):
    new_logits = []
    for logit in logits:
        token = logit[0]
        token = token.strip()
        if token in string.punctuation:
            continue
        else:
            if token.startswith("'"):
                token = token[1:]
            elif token == "":
                continue
            elif token.lower() not in stopword_ls:
                new_logits.append(logit)
    return new_logits


def filter_only_number(logits):
    new_logits = []
    for logit in logits:
        token = logit[0]
        token = token.strip()
        tokens = re.findall(r'[-+]?\d*\.\d+|\d+', token)
        if len(tokens) == 0:
            continue
        else:
            new_logits.append(logit)
    return new_logits


def base_or_persona_with_filtering(args, question, response, stopword_ls, model, tokenizer, device):
    user_msg = question

    chat = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": response}
    ]
    if args.method == "role_play":
        persona_chat = [
            {"role": "user", "content": args.role_setting},
            {"role": "assistant", "content": args.reply},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response}
        ]
    else:
        persona_chat = [
            {"role": "system", "content": f"You are a {args.job}."},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response}
        ]
    base_token, base_logits = generate_next_step(
        chat, model, tokenizer, device)
    persona_token, persona_logits = generate_next_step(
        persona_chat, model, tokenizer, device)
    return_token = None
    flag = None
    base_gap = 0
    persona_gap = 0
    if args.stopword_type == 'stopword':
        base_logits = filter_stopword(base_logits, stopword_ls)
        persona_logits = filter_stopword(persona_logits, stopword_ls)
    else:
        base_logits = filter_only_number(base_logits)
        persona_logits = filter_only_number(persona_logits)

    if base_token == persona_token:
        response += base_token
        return_token = base_token
        flag = 0
    else:
        base_gap = np.mean([logit[1] for logit in base_logits])
        persona_gap = np.mean([logit[1] for logit in persona_logits])
        if base_gap > persona_gap:
            response += base_token
            return_token = base_token
            flag = 1
        else:
            response += persona_token
            return_token = persona_token
            flag = 2
    return response, return_token, flag, (base_gap, persona_gap)


def run_base_or_persona_with_filtering(args, question, stopword_ls, model, tokenizer, device, max_token=512):
    response = ""
    count = [0, 0, 0]
    dnl_list = []
    cnt = 0
    while cnt <= max_token:
        response, last_token, flag, gap = base_or_persona_with_filtering(
            args, question, response, stopword_ls, model, tokenizer, device)
        with open(f"./current_{args.dataset}_dnl_filtered.txt", "w") as f:
            f.write(
                f"Question: {question}\n\nresponse: {response}\n\nlast_token: {last_token}\n\ncnt: {cnt} / 512")
        count[flag] += 1
        dnl_list.append((last_token, flag, gap))
        if last_token == '':
            break
        cnt = len(tokenizer.encode(response)) - 1
    return response, count, dnl_list


def main():
    args = parse_arguments()
    model_name = args.model_name.replace('/', '_')
    print('*****************************')
    print(args)
    print('*****************************')

    fix_seed(args.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "| trim", "")  # contain \n\n in template
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, token=hf_token)
    device = 'cuda'
    model = model.to(device)

    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    if args.filter_stopword:
        nltk.download('stopwords')
        stopword_ls = stopwords.words('english')
        if '\n\n' not in stopword_ls:
            stopword_ls.append('\n\n')

    total = 0
    correct_list = []
    if not os.path.exists('./logs_new_decoding'):
        os.mkdir('./logs_new_decoding')

    for i, data in enumerate(dataloader):
        print('*************************')
        print("{}st data".format(i+1))

        # Prepare question template ...
        x, y = data

        x = x[0]
        y = y[0].strip()

        if args.filter_stopword:
            assert stopword_ls is not None
            response, count, dnl_list = run_base_or_persona_with_filtering(
                args, x, stopword_ls, model, tokenizer, device)
        else:
            response, count, dnl_list = run_base_or_persona(
                args, x, model, tokenizer, device)
        chat = [
            {"role": "user", "content": x},
            {"role": "assistant", "content": response},
            {"role": "user", "content": args.direct_answer_trigger.strip()}
        ]

        pred, _ = generation(chat, model, tokenizer, device)
        print(pred)
        pred = answer_cleansing(args, pred)
        pred = clean_pred(pred)

        # Choose the most frequent answer from the list ...
        print("pred : {}".format(pred))
        print("GT : " + y)
        print('*************************')

        # We clean the pred and GT here!
        # pred = clean_pred(pred)
        y = clean_ans(y)

        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        logfile = f'./logs_new_decoding/{model_name}_{args.dataset}_dnl_new_decoding{args.etc}'
        if args.method == 'role_play':
            logfile += "_fixed_persona"
        if args.filter_stopword:
            logfile += "_filtered_stopwords"
            if args.stopword_type == 'number':
                logfile += "_number"
        if args.cmp != 'big':
            logfile += f'_{args.cmp}'
            if args.cmp == 'random':
                logfile += f'_seed:{args.random_seed}'
        if args.compare == 'probs':
            logfile += "_probs"
        if args.normalization:
            logfile += '_normalized'

        with open(logfile+'.jsonl', "a+") as f:
            inst = {
                'idx': total,
                'question': x,
                'label': y,
                'pred': pred,
                'response': response,
                'count': count,
                'dnl_list': dnl_list
            }
            line = json.dumps(inst, ensure_ascii=False)
            f.write(line+'\n')
        total += 1  # np.array([y]).size(0)
        if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            break
            # raise ValueError("Stop !!")

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))


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

    parser.add_argument(
        "--filter_stopword", type=int, default=0, help="whether to filter stopwords during calculating confidence score for each steps"
    )
    parser.add_argument(
        "--stopword_type", type=str, choices=['stopword', 'number'], default='stopword', help="type of filtering method for filtering logits"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[
                        1])

    parser.add_argument("--max_num_worker", type=int, default=1,
                        help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="model name from Huggingface Hub"
    )

    parser.add_argument(
        "--method", type=str, default="role_play", choices=["zero_shot", "role_play"], help="method"
    )
    parser.add_argument(
        "--cmp", type=str, default="big", choices=['big', 'small', 'random'], help="selection method when comparing between base and persona's confidence scores"
    )
    parser.add_argument(
        "--compare", type=str, choices=['gaps', 'probs'], default='gaps', help="Confidence score to be used (gaps: logit gaps, probs: probabilities)"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        '--etc', type=str, default=""
    )
    parser.add_argument(
        "--normalization", type=int, default=0
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
