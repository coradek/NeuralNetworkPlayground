"""text_data.py
A place for demo text generation

"""
import numpy as np

def create_easy_token_extraction_data(data_size=1000,
                                      char_string="abcdefghijklmnopXY",
                                      # token_charstring="ghijkl",
                                      # separators=("[", "]")
                                      # prefix_charstring="abc",
                                      # suffix_charstring="def",
                                      prefix_len_range=(2, 8),
                                      suffix_len_range=(2, 8),
                                      token_len_range=(6, 24),
                                      ):
    """create_easy_token_extraction_data
    create a dataset where a token demarcated by special characters
    is imbedded in a larger string.

    :param data_size: size of the dataset
    :param char_string: string of chars - last two used to mark token
    :param prefix_len_range (tuple): range of possible lengths for the
        first section of the string
    :param suffix_len_range (tuple): range of possible lengths for the
        last section of the string
    :param prefix_len_range (tuple): range of possible lengths for the
        token to be extracted
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

        start_len = np.random.randint(*prefix_len_range)
        start_text = np.random.choice(char_regular, start_len)
        start_text = "".join(start_text)

        token_len = np.random.randint(*token_len_range)
        token = np.random.choice(char_regular, token_len)
        token = "".join(token)

        end_len = np.random.randint(*suffix_len_range)
        end_text = np.random.choice(char_regular, end_len)
        end_text = "".join(end_text)
        text =   start_text \
               + char_special[0] \
               + token \
               + char_special[1] \
               + end_text
        return np.array([text, token])

    sample_data = np.vstack((get_one_sample() for _ in range(data_size)))
    return sample_data


def create_addition_string_data(data_size=1000, addend_range=(0, 500),
                                # second_addend_range=None,
                                ):
    ## Third column as place holder
    addends = np.random.randint(*addend_range, (data_size, 3))
    ## set third column to sum of first two
    addends[:, 2] = addends[:, :2].sum(axis=1)
    def stringify(xx):
        return "{} + {}".format(*xx[:2]), str(xx[2])
    data = np.apply_along_axis(tempf, 1, temp_addends)
    return data


if __name__ == '__main__':
    sample = create_easy_token_extraction_data(data_size=5)
    print(sample)
