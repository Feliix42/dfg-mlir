#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <omp.h>

typedef int64_t chanTy;
struct chan {
    chanTy* items;
    int64_t head_idx;
    int64_t tail_idx;
    int64_t capacity;
    _Atomic int64_t occupancy;
    _Atomic bool connected;
};

struct result {
    chanTy result;
    bool success;
};

struct chan* channel_i64() {
    // TODO: null check
    struct chan* channel = malloc(sizeof(struct chan));
    chanTy* buffer = malloc(sizeof(chanTy) * 50);

    channel->items = buffer;
    channel->head_idx = 0;
    channel->tail_idx = 0;
    channel->capacity = 50;
    /*channel->occupancy = 0;*/
    /*channel->connected = 1;*/
    atomic_init(&(channel->occupancy), 0);
    atomic_init(&(channel->connected), true);
    return channel;
}

bool push_i64(struct chan* sender, chanTy to_send) {
    while (true) {
        if (!atomic_load_explicit(&(sender->connected), memory_order_relaxed))
            return false;

        int64_t occupancy = atomic_load(&(sender->occupancy));
        if (occupancy < sender->capacity) {
            sender->items[sender->tail_idx] = to_send;
            sender->tail_idx = (sender->tail_idx + 1) % sender->capacity;

            // atomic +1 size
            atomic_fetch_add(&(sender->occupancy), 1);

            return true;
        } else {
#pragma omp taskyield
            continue;
        }
    }
}

struct result pull_i64(struct chan* sender) {
    while (true) {
        if (atomic_load(&(sender->occupancy)) == 0) {
            // empty queue -> still active?
            if (!atomic_load_explicit(&(sender->connected), memory_order_relaxed)) {
                struct result failure;
                failure.success = false;
                return failure;
            }

            // if the queue is not dead, just wait
#pragma omp taskyield
            continue;
        } else {
            // non-empty
            chanTy elem = sender->items[sender->head_idx];
            sender->head_idx = (sender->head_idx + 1) % sender->capacity;

            atomic_fetch_sub(&(sender->occupancy), 1); 
            
            struct result out = {elem, true};
            return out;
        }
    }
}

void close_i64(struct chan* sender) {
    bool expected = true;
    // deallocate if this is the last remaining reference
    if (!atomic_compare_exchange_strong(&sender->connected, &expected, false)) {
        free(sender->items);
        free(sender);
    }

    return;
}























