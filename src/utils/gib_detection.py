"""The MIT License (MIT)

Copyright (c) 2015 Rob Renaud

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE."""

import math


class GibDetector:
    def __init__(self):
        self.accepted_chars = "abcdefghijklmnopqrstuvwxyz1234567890-."

        self.pos = dict([(char, idx) for idx, char in enumerate(self.accepted_chars)])

    def _normalize(self, line):
        """Return only the subset of chars from accepted_chars.
        This helps keep the  model relatively small by ignoring punctuation,
        infrequenty symbols, etc."""
        return [c.lower() for c in line if c.lower() in self.accepted_chars]

    def _ngram(self, n, l):
        """Return all n grams from l after normalizing"""
        filtered = self._normalize(l)
        for start in range(0, len(filtered) - n + 1):
            yield "".join(filtered[start : start + n])

    def train(
        self,
        training_data_path="../utils/.gib_detector_trainingdata.txt",
        good_examples_path="../utils/.gib_detector_goodexamples.txt",
        bad_examples_path="../utils/.gib_detector_badexamples.txt",
    ):
        """Write a simple model as a pickle file"""
        k = len(self.accepted_chars)
        # Assume we have seen 10 of each character pair.  This acts as a kind of
        # prior or smoothing factor.  This way, if we see a character transition
        # live that we've never observed in the past, we won't assume the entire
        # string has 0 probability.
        counts = [[10 for i in range(k)] for i in range(k)]

        # Count transitions from big text file, taken
        # from http://norvig.com/spell-correct.html
        for line in open(training_data_path):
            for a, b in self._ngram(2, line):
                counts[self.pos[a]][self.pos[b]] += 1

        # Normalize the counts so that they become log probabilities.
        # We use log probabilities rather than straight probabilities to avoid
        # numeric underflow issues with long texts.
        # This contains a justification:
        # http://squarecog.wordpress.com/2009/01/10/dealing-with-underflow-in-joint-probability-calculations/
        for i, row in enumerate(counts):
            s = float(sum(row))
            for j in range(len(row)):
                row[j] = math.log(row[j] / s)

        # Find the probability of generating a few arbitrarily choosen good and
        # bad phrases.
        good_probs = [
            self._avg_transition_prob(l, counts) for l in open(good_examples_path)
        ]
        bad_probs = [
            self._avg_transition_prob(l, counts) for l in open(bad_examples_path)
        ]

        # print(good_probs)
        # print(bad_probs)

        # # Assert that we actually are capable of detecting the junk.
        # assert min(good_probs) > max(bad_probs)

        # And pick a threshold halfway between the worst good and best bad inputs.
        # thresh = (min(good_probs) + max(bad_probs)) / 2
        self.thresh = 0.012
        self.counts = counts
        # pickle.dump({"mat": counts, "thresh": thresh}, open("gib_model.pkl", "wb"))

    def _avg_transition_prob(self, l, log_prob_mat):
        """Return the average transition prob from l through log_prob_mat."""
        log_prob = 0.0
        transition_ct = 0
        for a, b in self._ngram(2, l):
            log_prob += log_prob_mat[self.pos[a]][self.pos[b]]
            transition_ct += 1
        # The exponentiation translates from log probs to probs.
        return math.exp(log_prob / (transition_ct or 1))

    def detect(self, l):
        return self._avg_transition_prob(l, self.counts) > self.thresh
