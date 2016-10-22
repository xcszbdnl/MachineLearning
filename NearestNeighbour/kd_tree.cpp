/*
**********************************************
The AC code for
http://acm.hdu.edu.cn/showproblem.php?pid=2966
find nearest neighbour for each point
**********************************************
*/

#include <cstdio>
#include <algorithm>

using namespace std;

struct Point {
    int coor[2];
    int id;
};

const int maxn = 100007;
const int DIMENSION = 2;
struct Point point[maxn];
int point_num, cmp_dimension;
long long min_dist, ans[maxn];

int cmp(struct Point a, struct Point b) {
    return a.coor[cmp_dimension] < b.coor[cmp_dimension];
}

long long dist(struct Point &a, struct Point &b) {
    long long x = a.coor[0] - b.coor[0];
    long long y = a.coor[1] - b.coor[1];
    x *= x;
    y *= y;
    x += y;
    return x == 0 ? -1 : x;
}

void build_tree(int low, int high, int depth) {
    if (low >= high) {
        return ;
    }
    int mid = (low + high) >> 1;
    cmp_dimension = depth % DIMENSION;
    nth_element(point + low, point + mid, point + high + 1, cmp);
    build_tree(low, mid - 1, depth + 1);
    build_tree(mid + 1, high, depth + 1);
}

void find_nearest(struct Point &search_point, int low, int high, int depth) {
    if (low > high) {
        return ;
    }
    int mid = (low + high) >> 1;
    int cnt_dimension = depth % DIMENSION;
    long long cnt_dist = dist(search_point, point[mid]);
    if (cnt_dist != -1 && (min_dist == -1 || min_dist > cnt_dist)) {
        min_dist = cnt_dist;
    }
    if (search_point.coor[cnt_dimension] < point[mid].coor[cnt_dimension]) {
        find_nearest(search_point, low, mid - 1, depth + 1);
        long long d = point[mid].coor[cnt_dimension] - search_point.coor[cnt_dimension];
        d *= d;
        if (d < min_dist) {
            find_nearest(search_point, mid + 1, high, depth + 1);
        }
    }
    else {
        find_nearest(search_point, mid + 1, high, depth + 1);
        long long d = search_point.coor[cnt_dimension] - point[mid].coor[cnt_dimension];
        d *= d;
        if (d < min_dist) {
            find_nearest(search_point, low, mid - 1, depth + 1);
        }
    }
}


int main() {
//    freopen("in.txt", "r", stdin);
    int test_case;
    scanf("%d", &test_case);
    while (test_case--) {
        scanf("%d", &point_num);
        for (int i = 1; i <= point_num; i++) {
            scanf("%d%d", &point[i].coor[0], &point[i].coor[1]);
            point[i].id = i;
        }
        cmp_dimension = 0;
        build_tree(1, point_num, 0);
        for (int i = 1; i <= point_num; i++) {
            min_dist = -1;
            find_nearest(point[i], 1, point_num, 0);
            ans[point[i].id] = min_dist;
        }
        for (int i = 1; i <= point_num; i++) {
            printf("%I64d\n", ans[i]);
        }
    }
    return 0;
}
