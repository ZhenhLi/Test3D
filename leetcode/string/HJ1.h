/**
题目描述：
计算字符串最后一个单词的长度，单词以空格隔开，字符串长度小于5000。

输入描述：
输入一行，代表要计算的字符串，非空，长度小于5000。

输出描述：
输出一个整数，表示输入字符串最后一个单词的长度。

示例：
输入：hello nowcoder

输出：8
                        
原文链接：https://blog.csdn.net/zhaitianbao/article/details/118787331
*/
#include <iostream>
#include <cstring>

namespace lc{

int fun1() {
  // 解题思路：要输出最后一个单词，就倒着统计，遇到空格就停止统计；
  //         若只有一个单词，还要判断是否完成了整个字符串的遍历，如果全遍历都没碰到空格，则只有一个单词；
  
  char s[5001];
  std::cin.getline(s, 5001);

  int number = 0;
  int snumber = strlen(s);
  
  for (int i = snumber - 1; s[i] != ' '; --i) {
    number++;
    if (i == 0)
      break;
  }
  std::cout << number << std::endl;

  return 0;
}

}
