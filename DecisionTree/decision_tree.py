import matplotlib.pyplot as plt
import math
import operator


class DecsionNode:
    def __init__(self, results=None, children_list=None, split_value=None, split_feature=None):
        self.results = results
        self.split_feature = split_feature
        self.split_value = split_value
        self.children_list = children_list


def entropy(train_data):
    num_cases = len(train_data)
    count = {}
    results = 0
    for i in range(num_cases):
        label = train_data[i][-1]
        if label not in count.keys():
            count[label] = 0
        count[label] += 1
    for label in count.keys():
        prob = count[label] * 1.0 / num_cases
        results -= prob * math.log(prob, 2)
    return results


def create_data():
    train_data=[['slashdot', 'USA', 'yes', 18, 'None'],
        ['google', 'France', 'yes', 23, 'Premium'],
        ['digg', 'USA', 'yes', 24, 'Basic'],
        ['kiwitobes', 'France', 'yes', 23, 'Basic'],
        ['google', 'UK', 'no', 21, 'Premium'],
        ['(direct)', 'New Zealand', 'no', 12, 'None'],
        ['(direct)', 'UK', 'no', 21, 'Basic'],
        ['google', 'USA', 'no', 24, 'Premium'],
        ['slashdot', 'France', 'yes', 19, 'None'],
        ['digg', 'USA', 'no', 18, 'None'],
        ['google', 'UK', 'no', 18, 'None'],
        ['kiwitobes', 'UK', 'no', 19, 'None'],
        ['digg', 'New Zealand', 'yes', 12, 'Basic'],
        ['slashdot', 'UK', 'no', 21, 'None'],
        ['google', 'UK', 'yes', 18, 'Basic'],
        ['kiwitobes', 'France', 'yes', 19, 'Basic']]
    attribute_name = ['Referer', 'Country', 'Read FAQ', '# of webpages visited']
    return train_data, attribute_name


def split_data(train_data, axis, value):
    return_data = []
    for cnt_data in train_data:
        if cnt_data[axis] == value:
            cut_data = cnt_data[:axis]
            cut_data.extend(cnt_data[axis+1:])
            return_data.append(cut_data)
    return return_data


def get_best_feature(train_data, feature_name):
    best_index = 0
    best_information_gain = 0.0
    data_entropy = entropy(train_data)
    attribute_len = len(feature_name)
    num_cases = len(train_data)
    for i in range(attribute_len):
        cnt_features = [example[i] for example in train_data]
        unique_features = set(cnt_features)
        split_entropy = 0.0
        for feature in unique_features:
            feature_subset = split_data(train_data, i, feature)
            prob = len(feature_subset) / num_cases
            split_entropy += prob * entropy(feature_subset)
        cnt_gain = data_entropy - split_entropy
        if cnt_gain > best_information_gain:
            best_information_gain = cnt_gain
            best_index = i
    return best_index


def major_count(train_data):
    label_count = {}
    for cnt_data in train_data:
        label = cnt_data[-1]
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1
    sorted_result = sorted(label_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_result[0][0]


def build_tree(train_data, feature_name):
    class_list = [example[-1] for example in train_data]
    if class_list.count(class_list[0]) == len(class_list):
        return DecsionNode(results=class_list[0])
    if len(feature_name) == 0:
        label = major_count(train_data)
        return DecsionNode(results=label)
    feature_index = get_best_feature(train_data, feature_name)
    cnt_node = DecsionNode(split_feature=feature_name[feature_index], children_list=[], split_value=[])
    del(feature_name[feature_index])
    feature_values = [example[feature_index] for example in train_data]
    unique_values = set(feature_values)
    for feature in unique_values:
        sub_feature = feature_name[:]
        child_node = build_tree(split_data(train_data, feature_index, feature), sub_feature)
        cnt_node.children_list.append(child_node)
        cnt_node.split_value.append(feature)
    return cnt_node


def get_leaf_num(tree):
    if tree.results is not None:
        return 1
    else:
        leaf_num = 0
        for node in tree.children_list:
            leaf_num += get_leaf_num(node)
        return leaf_num


def get_depth(tree):
    if tree.results is not None:
        return 1
    else:
        max_depth = 0
        for node in tree.children_list:
            cnt_depth = get_depth(node)
            if cnt_depth > max_depth:
                max_depth = cnt_depth
        return max_depth + 1


DECISION_DRAW = dict(boxstyle='sawtooth', fc='0.8')
LEAF_DRAW = dict(boxstyle='round4', fc='0.8')
ARROW_DRAW = dict(arrowstyle='<-')


def plot_node(node_txt, cnt_point, parent_point, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_point, xycoords='axes fraction',
                             xytext=cnt_point, textcoords='axes fraction',
                             va='center', ha='center', bbox=node_type, arrowprops=ARROW_DRAW)


def plot_mid_text(cnt_point, parent_point, txt_str):
    xmid = (parent_point[0] - cnt_point[0]) / 2.0 + cnt_point[0]
    ymid = (parent_point[1] - cnt_point[1]) / 2.0 + cnt_point[1]
    create_plot.ax1.text(xmid, ymid, txt_str)


def plot_tree(tree, parent_point, node_txt):
    leaf_num = get_leaf_num(tree)
    #depth = get_depth(tree)
    first_str = tree.split_feature
    cnt_point = (plot_tree.xOff + (1.0 + float(leaf_num)) / 2.0 / plot_tree.total_w, plot_tree.yOff)
    plot_mid_text(cnt_point, parent_point, node_txt)
    plot_node(first_str, cnt_point, parent_point, DECISION_DRAW)
    plot_tree.yOff -= 1.0 / plot_tree.total_d
    children_len = len(tree.children_list)
    for i in range(children_len):
        node = tree.children_list[i]
        if node.results is None:
            plot_tree(node, cnt_point, str(tree.split_value[i]))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.total_w
            plot_node(str(node.results), (plot_tree.xOff, plot_tree.yOff), cnt_point, LEAF_DRAW)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cnt_point, str(tree.split_value[i]))
    plot_tree.yOff += 1.0 / plot_tree.total_d


def create_plot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprobs = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprobs)
    plot_tree.total_w = float(get_leaf_num(tree))
    plot_tree.total_d = float(get_depth(tree))
    plot_tree.xOff = -0.5 / plot_tree.total_w
    plot_tree.yOff = 1.0
    plot_tree(tree, (0.5, 1.0), '')
    plt.show()


def decision_tree():
    train_data, attribute_name = create_data()
    final_tree = build_tree(train_data, attribute_name)
    create_plot(final_tree)


if __name__ == '__main__':
    decision_tree()