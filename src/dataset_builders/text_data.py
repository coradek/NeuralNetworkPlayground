"""text_data.py
A place for demo text generation

"""
import numpy as np

def create_easy_token_extraction_data(data_size=1000,
                                      char_string="abcdefghijklmnopXY"):
    """create_easy_token_extraction_data
    :param data_size: size of the dataset
    :param char_string: string of chars - last two used to mark token
    :return: dataset - text, token pairs
    [["aaaaXtokenYbbb", "token"]]
    """
    def get_one_sample(char_string=char_string):
        """
        :param char_string: string of chars - last two used to mark token
        :return: datapoint - text, token pair
        """

        char_regular = np.array(list(char_string[:-2]))
        char_special = np.array(list(char_string[-2:]))

        start_len = np.random.randint(2, 12)
        start_text = np.random.choice(char_regular, start_len)
        start_text = "".join(start_text)

        token_len = np.random.randint(2, 8)
        token = np.random.choice(char_regular, token_len)
        token = "".join(token)

        end_len = np.random.randint(6, 24)
        end_text = np.random.choice(char_regular, end_len)
        end_text = "".join(end_text)

        text =   start_text \
               + char_special[0] \
               + token \
               + char_special[1] \
               + end_text

        return np.array([text, token])

    sample_data = np.vstack((get_one_sample() for _ in range(data_size)))
    # sample = get_one_sample()
    return sample_data


if __name__ == '__main__':
    sample = create_easy_token_extraction_data(data_size=5)
    print(sample)
