/*
题目描述：
写出一个程序，接受一个由字母、数字和空格组成的字符串，和一个字母，然后输出输入字符串中该字母的出现次数。不区分大小写，字符串长度小于500。

输入描述：
第一行输入一个由字母和数字以及空格组成的字符串，第二行输入一个字母。

输出描述：
输出输入字符串中含有该字符的个数。

示例:
输入：

ABCabc

A

输出：

2

原文链接：https://blog.csdn.net/zhaitianbao/article/details/118787463
*/

#include <iostream>
#include <cstring>

namespace lc{

void fun_hj2() {
  // 遍历字符串、
  // 将每个字符和输入的字符转换为小写(或大写)直接进行比较
  // 如果相等，计数自增

  char s[500];
  std::cin.getline(s, 500);

  int snumber = strlen(s);

  char test;
  std::cin >> test;

  int number = 0;
  for (int i = 0; i < snumber; i++) {
    if (tolower(s[i]) == tolower(test)) { // https://blog.csdn.net/tax10240809163com/article/details/117882087
      number++;
    }
  }

  std::cout << number << std::endl;

  return ;
}

}