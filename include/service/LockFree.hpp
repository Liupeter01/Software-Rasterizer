#pragma once
#ifndef _LOCKFREE_HPP_ 
#define _LOCKFREE_HPP_ 
#include <thread>
#include <atomic>

struct SpinLock {
          void lock() { 
                    while (_flag.test_and_set(std::memory_order_acquire)) {
                              std::this_thread::yield();
                    }
          }

          void unlock() { 
                    _flag.clear(std::memory_order_release); 
          }

private:
          std::atomic_flag _flag = ATOMIC_FLAG_INIT;
};

#endif //_LOCKFREE_HPP_ 
