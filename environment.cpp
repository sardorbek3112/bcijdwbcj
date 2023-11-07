#include <bits/stdc++.h>

using namespace std;
int prime(int n){
    vector<int>v(10000,1);
    int l = 10000;
    for (int i = 2;i<l;i++){
        if (v[i] == 1){
            for (int j = i*i;j<l;j += i){
                v[j] = 0;
            }
        }
    }
    int o = 0;
    for (int i = 2;i<l;i++){
        if (v[i] == 1) o ++;
        if (o == n) return i;
    }
}
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    vector<int> a= {1,2,3};
    binary_search(a.begin(),a.end(),3)
    int n = 4;
    cout<<prime(n);
}
