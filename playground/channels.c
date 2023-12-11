#include <omp.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*#include <stdio.h>*/

struct chan {
    char* items;
    uint64_t head_idx;
    uint64_t tail_idx;
    uint64_t bytewidth;
    uint64_t capacity;
    _Atomic uint64_t occupancy;
    _Atomic bool connected;
};

struct chan* channel(uint64_t bytewidth)
{
    /*printf("byte width: %ld\n", bytewidth);*/
    // TODO: null check
    struct chan* channel = (struct chan*)malloc(sizeof(struct chan));
    char* buffer = (char*)malloc(bytewidth * 50);

    channel->items = buffer;
    channel->head_idx = 0;
    channel->tail_idx = 0;
    channel->bytewidth = bytewidth;
    channel->capacity = 50;
    atomic_init(&(channel->occupancy), 0);
    atomic_init(&(channel->connected), true);
    return channel;
}

bool push(struct chan* sender, char* to_send)
{
    /*printf("Beginning Send\n");*/
    while (true) {
        if (!atomic_load_explicit(&(sender->connected), memory_order_relaxed))
            return false;

        uint64_t occupancy = atomic_load(&(sender->occupancy));
        if (occupancy < sender->capacity) {
            memcpy(
                (sender->items + (sender->tail_idx * sender->bytewidth)),
                to_send,
                sender->bytewidth);
            sender->tail_idx = (sender->tail_idx + 1) % sender->capacity;

            // atomic +1 size
            atomic_fetch_add(&(sender->occupancy), 1);

            return true;
        } else {
/*#pragma omp taskyield*/
            continue;
        }
    }
}

bool pull(struct chan* recv, char* result)
{
    /*printf("Beginning pull\n");*/
    while (true) {
        if (atomic_load(&(recv->occupancy)) == 0) {
            // empty queue -> still active?
            if (!atomic_load_explicit(&(recv->connected), memory_order_relaxed))
                return false;

                // if the queue is not dead, just wait
/*#pragma omp taskyield*/
            continue;
        } else {
            // non-empty
            memcpy(
                result,
                (recv->items + (recv->head_idx * recv->bytewidth)),
                recv->bytewidth);
            recv->head_idx = (recv->head_idx + 1) % recv->capacity;

            atomic_fetch_sub(&(recv->occupancy), 1);

            return true;
        }
    }
}

void close_channel(struct chan* sender)
{
    if (!sender)
        return;
    /*printf("be gone\n");*/
    bool expected = true;
    // deallocate if this is the last remaining reference
    if (!atomic_compare_exchange_strong(&(sender->connected), &expected, false)) {
        /*printf("de-alloc shimashou!\n");*/
        free(sender->items);
        free(sender);
    }

    return;
}
