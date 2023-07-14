import multiprocessing
from itertools import product

import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

np.seterr(invalid='ignore')

indicators = np.empty(0)
question_answer_to_participants = dict()
num_answers = 0
cur_partition = []
def scaled_mean_entropy(subset):
    subset_counts = indicators[:, list(subset), :].sum(axis=1)
    ent = entropy(subset_counts, axis=0, base=2)
    remove_nans = np.nan_to_num(ent)
    mean_ent = remove_nans.mean()
    scaled = mean_ent * len(subset)
    return scaled


def compute_finer_partition(question):
    return [subset.intersection(question_answer_to_participants[question, answer]) for
            answer in range(num_answers) for subset in cur_partition]


def question_entropy(question):
    finer_partition = compute_finer_partition(question)
    return sum(scaled_mean_entropy(subset) for subset in finer_partition)


def greedy_k_entropy(k, data, total_num_answers=3):
    global num_answers
    global indicators
    global question_answer_to_participants
    global cur_partition

    num_answers = total_num_answers

    data_np = data.to_numpy()

    num_participants, num_questions = data_np.shape
    # indicators[i, j, k] = 1 iff user j answered i to question k.
    indicators = np.stack([(data_np == answer) for answer in range(num_answers)])
    # question_answer_to_participants[q, a] = set of users that answered a to question q
    question_answer_to_participants = {
        (question, answer): set(np.nonzero(indicators[answer, :, question])[0]) for question, answer in
        product(range(num_questions), range(num_answers))
    }

    selected_questions = []
    remaining_questions = list(range(num_questions))

    cur_partition = [set(range(num_participants))]

    for iteration in range(k):
        with multiprocessing.get_context('fork').Pool() as p:
            cur_question_entropies = list(tqdm(p.imap(question_entropy, remaining_questions), total=len(remaining_questions), desc=f'{iteration + 1}'))
        question_to_entropy = dict(zip(remaining_questions, cur_question_entropies))

        best_question = min(question_to_entropy, key=question_to_entropy.get)

        print(f'{iteration + 1:2}: {data.columns[best_question]:10} {question_to_entropy[best_question] / len(data)}')

        selected_questions.append(best_question)
        remaining_questions.remove(best_question)
        new_partition = compute_finer_partition(best_question)
        cur_partition = [s for s in new_partition if len(s) > 1]  # Remove trivial partition elements
    return selected_questions
