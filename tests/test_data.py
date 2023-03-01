import unittest
import numpy as np
import subprocess
import os
import pickle
import pdb


def make_dummy_data():
    num_sentences = np.random.randint(10, 100)
    num_labels = np.random.randint(0, 10)
    sentences = [[str(syl) for syl in np.random.randint(0, 100, size=np.random.randint(10, 100))] for _ in range(num_sentences)]
    labels = np.random.randint(0, num_labels, size=num_sentences)
    return sentences, labels


class DataTest(unittest.TestCase):
    def test_phrases(self):
        sentences, labels = make_dummy_data()

        sentences_fn = "./sentences.pkl"
        labels_fn = "./labels.pkl"

        for nm, dt in zip([sentences_fn, labels_fn], [sentences, labels]):
            with open(nm, "wb") as fn:
                pickle.dump(dt, fn)

        data_path = "."
        save_dir = "./phrases"
        emissions = [True, False]
        scoring = ["default", "npmi"]

        for em in emissions:
            for sc in scoring:
                iter_dir = os.path.join(save_dir, f"phrase_iterations_1")
                process_string = f"moseq2-nlp make-phrases {data_path} --save-dir {save_dir} --scoring {sc}"
                if em:
                    process_string += " --emissions"
                subprocess.call(process_string, shell=True)
                self.assertTrue(os.path.exists(os.path.join(iter_dir, "sentences.pkl")))
                subprocess.call(f"rm -rf {save_dir}", shell=True)
        subprocess.call(f"rm ./sentences.pkl ./labels.pkl", shell=True)

    def test_synonyms(self):
        sentences, labels = make_dummy_data()

        sentences_fn = "./sentences.pkl"
        labels_fn = "./labels.pkl"

        for nm, dt in zip([sentences_fn, labels_fn], [sentences, labels]):
            with open(nm, "wb") as fn:
                pickle.dump(dt, fn)

        data_path = "."
        save_dir = "./synonyms"
        emissions = [True, False]

        for em in emissions:
            process_string = f"moseq2-nlp make-synonyms {data_path} {save_dir}"
            if em:
                process_string += " --emissions"
            subprocess.call(process_string, shell=True)
            for dr in os.listdir(save_dir):
                self.assertTrue(os.path.exists(os.path.join(save_dir, dr, "sentences.pkl")))
                self.assertTrue(os.path.exists(os.path.join(save_dir, dr, "labels.pkl")))
            subprocess.call(f"rm -rf {save_dir}", shell=True)
        subprocess.call(f"rm ./sentences.pkl ./labels.pkl", shell=True)


if __name__ == "__main__":
    unittest.main()
