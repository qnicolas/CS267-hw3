#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <upcxx/upcxx.hpp>
#include <vector>

#include "hash_map.hpp"
#include "kmer_t.hpp"
#include "read_kmers.hpp"

#include "butil.hpp"

int main(int argc, char** argv) {
    upcxx::init();

    if (argc < 2) {
        BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test [prefix]]\n");
        upcxx::finalize();
        exit(1);
    }

    std::string kmer_fname = std::string(argv[1]);
    std::string run_type = "";

    if (argc >= 3) {
        run_type = std::string(argv[2]);
    }

    std::string test_prefix = "test";
    if (run_type == "test" && argc >= 4) {
        test_prefix = std::string(argv[3]);
    }

    int ks = kmer_size(kmer_fname);

    if (ks != KMER_LEN) {
        throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
                                 "-mers, while this binary is compiled for " +
                                 std::to_string(KMER_LEN) +
                                 "-mers.  Modify packing.hpp and recompile.");
    }

    size_t n_kmers = line_count(kmer_fname);

    // Load factor of 0.5
    size_t hash_table_size = n_kmers * (1.0 / 0.5);
    HashMap hashmap(hash_table_size);

    if (run_type == "verbose") {
        BUtil::print("Initializing hash table of size %d for %d kmers.\n", hash_table_size,
                     n_kmers);
    }

    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

    if (run_type == "verbose") {
        BUtil::print("Finished reading kmers.\n");
    }
    
    
    //////////////////////////////////////////////////////////// TESTS //////////////////////////////////////////////////////////
    //upcxx::dist_object<upcxx::global_ptr<int>>* used_pt; 
    //int slots_per_rank = hash_table_size/upcxx::rank_n()+1;
    //upcxx::dist_object<upcxx::global_ptr<int>> used1(upcxx::new_array<int>(slots_per_rank));
    //used_pt = &used1;
    //
    //upcxx::global_ptr<int> heads[upcxx::rank_n()];
    //for (int rank = 0; rank < upcxx::rank_n(); rank++){
    //    heads[rank] = used_pt->fetch(rank).wait();
    //}
    //
    //int test_int = upcxx::rget(heads[2]+10).wait();
    //BUtil::print("%i\n",test_int);
    //
    //BUtil::print("Finished program\n");
    //hashmap.ad.destroy();
    //return 0;
    

    auto start = std::chrono::high_resolution_clock::now();
    auto part_begin = std::chrono::high_resolution_clock::now();
    auto part_end = std::chrono::high_resolution_clock::now();
    double insert_time1 = 0.,insert_timemore = 0.;

    std::vector<kmer_pair> start_nodes;
    
    int all_counts = 0;
    int max_count = 0;
    
    for (auto& kmer : kmers) {
//        bool success = hashmap.insert(kmer);
//        if (!success) {
//            throw std::runtime_error("Error: HashMap is full!");
//        }
        part_begin = std::chrono::high_resolution_clock::now();
        int count = hashmap.insert(kmer);
        part_end = std::chrono::high_resolution_clock::now();
        
        all_counts+=count;
        max_count = max_count>=count ? max_count : count;
        switch (count) {
            case 0:
                throw std::runtime_error("Error: HashMap is full!");
                break;
            case 1:
                insert_time1 += std::chrono::duration<double>(part_end - part_begin).count();
                break;
            default:
                insert_timemore += std::chrono::duration<double>(part_end - part_begin).count();
        }
        

        if (kmer.backwardExt() == 'F') {
            start_nodes.push_back(kmer);
        }
    }
    auto end_insert = std::chrono::high_resolution_clock::now();
    upcxx::barrier();

    BUtil::print("mean count %f, max %i \n", ((double) all_counts) / ((double) n_kmers), max_count );
    BUtil::print("time1 %lf, timemore %lf\n", insert_time1,insert_timemore);
    
    double insert_time = std::chrono::duration<double>(end_insert - start).count();
    if (run_type != "test") {
        BUtil::print("Finished inserting in %lf\n", insert_time);
    }
    upcxx::barrier();
    
    
    
    
    hashmap.ad.destroy();
    upcxx::finalize();    
    return 0;
    
    
    

    auto start_read = std::chrono::high_resolution_clock::now();

    std::list<std::list<kmer_pair>> contigs;
    for (const auto& start_kmer : start_nodes) {
        std::list<kmer_pair> contig;
        contig.push_back(start_kmer);
        while (contig.back().forwardExt() != 'F') {
            kmer_pair kmer;
            bool success = hashmap.find(contig.back().next_kmer(), kmer);
            if (!success) {
                throw std::runtime_error("Error: k-mer not found in hashmap.");
            }
            contig.push_back(kmer);
        }
        contigs.push_back(contig);
    }

    auto end_read = std::chrono::high_resolution_clock::now();
    upcxx::barrier();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> read = end_read - start_read;
    std::chrono::duration<double> insert = end_insert - start;
    std::chrono::duration<double> total = end - start;

    int numKmers = std::accumulate(
        contigs.begin(), contigs.end(), 0,
        [](int sum, const std::list<kmer_pair>& contig) { return sum + contig.size(); });

    if (run_type != "test") {
        BUtil::print("Assembled in %lf total\n", total.count());
    }

    if (run_type == "verbose") {
        printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
               " (%lf read, %lf insert, %lf total)\n",
               upcxx::rank_me(), contigs.size(), numKmers, start_nodes.size(), read.count(),
               insert.count(), total.count());
    }

    if (run_type == "test") {
        std::ofstream fout(test_prefix + "_" + std::to_string(upcxx::rank_me()) + ".dat");
        for (const auto& contig : contigs) {
            fout << extract_contig(contig) << std::endl;
        }
        fout.close();
    }

    hashmap.ad.destroy();
    upcxx::finalize();
    return 0;
}
