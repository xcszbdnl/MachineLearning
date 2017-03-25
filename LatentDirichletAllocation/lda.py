import random

alpha = 0.1
beta = 0.1
k = 20
num_iter = 100
top_words = 5


class Document:
    def __init__(self):
        self.words = []
        self.length = 0


class DataSet:
    def __init__(self):
        self.document = []
        self.V = 0
        self.M = 0
        self.word_to_id = {}
        self.id_to_word = {}
        self.frequency = {}


class LDAModel:
    def __init__(self):
        self.alpha = alpha
        self.beta = beta
        self.K = k
        self.num_iter = num_iter
        self.top_words = top_words
        self.topic = []
        self.topic_word = []
        self.topic_word_sum = []
        self.topic_document = []
        self.word_sum = []
        self.theta = []
        self.phi = []
        self.dataset = None

    def init_param(self, dataset):
        self.dataset = dataset
        self.topic_word = [[0 for y in xrange(self.K)] for x in xrange(self.dataset.V)]
        self.topic_word_sum = [0 for x in xrange(self.K)]
        self.topic_document = [[0 for y in xrange(self.K)] for x in xrange(self.dataset.M)]
        self.word_sum = [0 for x in xrange(self.dataset.M)]
        self.topic = [[] for x in xrange(self.dataset.M)]
        for x in xrange(self.dataset.M):
            self.topic[x] = [0 for y in xrange(self.dataset.document[x].length)]
            self.word_sum[x] = self.dataset.document[x].length
            for y in xrange(self.word_sum[x]):
                cnt_topic = random.randint(0, self.K - 1)
                self.topic[x][y] = cnt_topic
                self.topic_word[self.dataset.document[x].words[y]][cnt_topic] += 1
                self.topic_document[x][cnt_topic] += 1
                self.topic_word_sum[cnt_topic] += 1
        self.theta = [[0.0 for y in xrange(self.K)] for x in xrange(self.dataset.M)]
        self.phi = [[0.0 for y in xrange(self.dataset.V)] for x in xrange(self.K)]

    def train(self):
        for x in xrange(self.num_iter):
            print 'iteration %d ...' % (x + 1)
            for i in xrange(self.dataset.M):
                for j in xrange(self.dataset.document[i].length):
                    sample_topic = self.sample(i, j)
                    self.topic[i][j] = sample_topic
        print 'sample finished'

        print 'compute theta'

        self.compute_theta()

        print 'compute theta finished'

        self.compute_phi()

        print 'compute phi finished'

        print 'get top words'

        self.get_top_words()

    def sample(self, i, j):
        pre_topic = self.topic[i][j]
        index_id = self.dataset.document[i].words[j]
        self.topic_word[index_id][pre_topic] -= 1
        self.topic_document[i][pre_topic] -= 1
        self.topic_word_sum[pre_topic] -= 1
        self.word_sum[i] -= 1

        beta_v = self.dataset.V * self.beta
        alpha_k = self.K * self.alpha
        temp_p = [0 for x in xrange(self.K)]
        for k in range(self.K):
            temp_p[k] = (self.topic_word[index_id][k] + self.beta) / (self.topic_word_sum[k] + beta_v) * \
                        (self.topic_document[i][k] + alpha) / (self.word_sum[i] + alpha_k)
        for k in range(1, self.K):
            temp_p[k] += temp_p[k - 1]

        u = random.uniform(0, temp_p[self.K-1])
        next_topic = 0
        for k in xrange(self.K):
            if temp_p[k] > u:
                next_topic = k
                break
        self.topic_word[index_id][next_topic] += 1
        self.topic_document[i][next_topic] += 1
        self.topic_word_sum[next_topic] += 1
        self.word_sum[i] += 1
        return next_topic

    def compute_theta(self):
        for x in xrange(self.dataset.M):
            for y in xrange(self.K):
                self.theta[x][y] = (self.topic_document[x][y] + self.alpha) / (self.word_sum[x] + self.K * self.alpha)

    def compute_phi(self):
        for x in xrange(self.K):
            for y in xrange(self.dataset.V):
                self.phi[x][y] = (self.topic_word[y][x] + self.beta) / (self.topic_word_sum[x] + self.dataset.V * self.beta)

    def get_top_words(self):
        for x in xrange(self.K):
            print 'topic k:'
            max_pro = -1.0
            words_index = 0
            for y in xrange(self.dataset.V):
                if self.phi[x][y] > max_pro:
                    max_pro = self.phi[x][y]
                    words_index = y
            print 'topic k top words is %s, pro is %f' % (self.dataset.id_to_word[words_index], max_pro)



def readData():
    dataset = DataSet()
    word_file = open('nips.vocab', 'r')

    for lines in word_file.readlines():
        [cnt_id, cnt_word, freq] = lines.split()
        if dataset.word_to_id.get(cnt_word) is not None:
            print('error\n')
            return
        cnt_id = int(cnt_id)
        dataset.word_to_id[cnt_word] = cnt_id
        dataset.id_to_word[cnt_id] = cnt_word

    text_file = open('nips.libsvm')

    for lines in text_file.readlines():
        cnt_doc = Document()
        split_result = lines.split()
        doc_id = int(split_result[0])
        split_result.remove(split_result[0])
        tot_length = 0
        word_counter_dict = {}
        for i in range(0, len(split_result)):
            id_count = split_result[i]
            [word_id, word_count] = id_count.split(':')
            word_id = int(word_id)
            word_count = int(word_count)
            for j in range(word_count):
                cnt_doc.words.append(word_id)
        cnt_doc.length = len(cnt_doc.words)
        dataset.document.append(cnt_doc)
    dataset.M = len(dataset.document)
    dataset.V = len(dataset.word_to_id)
    return dataset


def lda():
    dataset = readData()
    model = LDAModel()
    model.init_param(dataset)
    model.train()


if __name__ == '__main__':
    lda()
