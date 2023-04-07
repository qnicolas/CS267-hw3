#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

#include "butil.hpp"

struct HashMap {
    upcxx::global_ptr<kmer_pair>* data_pt;    
    upcxx::global_ptr<int>* used_pt; 
    upcxx::atomic_domain<int> ad; 
        
//    std::vector<kmer_pair> data;
//    std::vector<int> used;
    
    int slots_per_rank;

    size_t my_size;

    size_t size() const noexcept;

    HashMap(size_t size);

    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    int insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);
    
    
    
    // Helper functions

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
    upcxx::future<int> request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
};

HashMap::HashMap(size_t size) {
    my_size = size;
//    data.resize(size);
//    used.resize(size, 0);
    
    slots_per_rank = size/upcxx::rank_n()+1;
    BUtil::print("slots_per_rank is %i\n",slots_per_rank);
    
    upcxx::dist_object<upcxx::global_ptr<kmer_pair>> data(upcxx::new_array<kmer_pair>(slots_per_rank));
    upcxx::dist_object<upcxx::global_ptr<int>> used(upcxx::new_array<int>(slots_per_rank));
    
    data_pt = new upcxx::global_ptr<kmer_pair>[upcxx::rank_n()];
    used_pt = new upcxx::global_ptr<int>[upcxx::rank_n()];
    
    for (int rank = 0; rank < upcxx::rank_n(); rank++){
        data_pt[rank] = data.fetch(rank).wait();
        used_pt[rank] = used.fetch(rank).wait();
    }
    
    ad = upcxx::atomic_domain<int>({upcxx::atomic_op::load,upcxx::atomic_op::fetch_add});
}

//bool HashMap::insert(const kmer_pair& kmer) {
//    uint64_t hash = kmer.hash();
//    uint64_t probe = 0;
//    bool success = false;
//    do {
//        uint64_t slot = (hash + probe++) % size();
//        success = request_slot(slot).wait()>0 ? false : true;
//        // DO A RPC HERE !!
//        if (success) {
//            write_slot(slot, kmer);
//        }
//    } while (!success && probe < size());
//    return success;
//}

int HashMap::insert(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    int count = 0;
    do {
        count++;
        uint64_t slot = (hash + probe++) % size();
        // DO A RPC HERE !!
        success = request_slot(slot).wait()>0 ? false : true;
        if (success) {
            write_slot(slot, kmer);
        }
    } while ((!success) && probe < size());
    if (!success) count=0;
    return count;
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
            }
        }
    } while (!success && probe < size());
    return success;
}

bool HashMap::slot_used(uint64_t slot) { 
    //upcxx::global_ptr<int> remote_ar = used_pt->fetch(slot/slots_per_rank).wait(); ////// CAN AVOID THE WAIT() HERE - WITH A FUTURE - SEE ATOMICS SECTION OF THE GUIDE /////
    //upcxx::global_ptr<int> remote_slot = remote_ar + slot%slots_per_rank;
    upcxx::global_ptr<int> remote_slot = used_pt[slot/slots_per_rank] + slot%slots_per_rank;
    return ad.load(remote_slot,std::memory_order_relaxed).wait() != 0; 
}

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) {
    upcxx::global_ptr<kmer_pair> remote_kmer_slot = data_pt[slot/slots_per_rank] + slot%slots_per_rank;
    auto fut = upcxx::rput(kmer,remote_kmer_slot); ////// NO WAIT /////
}

kmer_pair HashMap::read_slot(uint64_t slot) {
    upcxx::global_ptr<kmer_pair> remote_kmer_slot = data_pt[slot/slots_per_rank] + slot%slots_per_rank;
    return upcxx::rget(remote_kmer_slot).wait();
}

upcxx::future<int> HashMap::request_slot(uint64_t slot) {
    upcxx::global_ptr<int> remote_slot = used_pt[slot/slots_per_rank] + slot%slots_per_rank;
    return ad.fetch_add(remote_slot, 1, std::memory_order_relaxed);
}

size_t HashMap::size() const noexcept { return my_size; }
