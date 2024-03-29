#include <torch/torch.h>
#include <iostream>

int main() {
  std::cout << "---------- test libtorch ----------" << std::endl;

  torch::Tensor tensor = torch::eye(3);
  torch::Tensor t = torch::rand({2, 3});
  std::cout << t << std::endl;
  return 0;
}

// 编译说明

// cmake -DCMAKE_PREFIX_PATH=/home/lzh/code/libtorch ..

// https://blog.csdn.net/john_bh/article/details/108221748