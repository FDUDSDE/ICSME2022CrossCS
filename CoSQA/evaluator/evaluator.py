import logging
import sys, json
import numpy as np


def read_answers(filename):
    answers = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            answers[js['url']] = js['idx']
    return answers


def read_predictions(filename):
    predictions = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            predictions.append(js)
    return predictions


def calculate_scores(predictions):
    scores = []
    for prediction in predictions:
        ans = prediction['ans']
        rank_list = prediction['rank']
        rank = rank_list.index(ans)
        scores.append(1/(rank+1))


    result = {}
    result['MRR'] = round(np.mean(scores), 4)

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for NL-code-search-Adv dataset.')
    # parser.add_argument('--answers', '-a', help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p', help="filename of the leaderboard predictions, in txt format.")

    args = parser.parse_args()
    # answers = read_answers(args.answers)
    predictions = read_predictions(args.predictions)
    scores = calculate_scores(predictions)
    print(scores)


if __name__ == '__main__':
    main()
