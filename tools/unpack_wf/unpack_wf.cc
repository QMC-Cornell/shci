#include <hps/src/hps.h>
#include <shci/src/det/det.h>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>

constexpr double SQRT2_INV = 0.7071067811865475;

class Wavefunction {
public:
  unsigned n_up = 0;

  unsigned n_dn = 0;

  double energy_hf = 0.0;

  std::vector<double> energy_var;

  bool time_sym = false;

  std::vector<Det> dets;

  std::vector<std::vector<double>> coefs;

  size_t get_n_dets() const { return dets.size(); }                                                                                                          
                                                                                                                                                             
  void unpack_time_sym() {                                                                                                                                   
    const size_t n_dets_old = get_n_dets();                                                                                                                  
    for (size_t i = 0; i < n_dets_old; i++) {                                                                                                                
      const auto& det = dets[i];                                                                                                                             
      if (det.up < det.dn) {                                                                                                                                 
        Det det_rev = det;                                                                                                                                   
        det_rev.reverse_spin();                                                                                                                              
        for (auto& state_coefs: coefs) {                                                                                                                     
          const double coef_new = state_coefs[i] * SQRT2_INV;                                                                                                
          state_coefs[i] = coef_new;                                                                                                                         
          state_coefs.push_back(coef_new);                                                                                                                   
        }                                                                                                                                                    
        dets.push_back(det_rev);                                                                                                                             
      }                                                                                                                                                      
    }                                                                                                                                                        
  }

  template <class B>
  void parse(B& buf) {
    buf >> n_up >> n_dn >> dets >> coefs >> energy_hf >> energy_var >> time_sym;
    if (time_sym) unpack_time_sym();
  }
};

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: exe wf_filename\n");
        return 1;
    }

    std::ifstream serialized_wf(argv[1], std::ios::binary); 
    Wavefunction wf = hps::from_stream<Wavefunction>(serialized_wf);

    //indices of sorted coefs/dets
    std::vector<size_t> inds(wf.dets.size());
    //fill with consecutive ints
    std::iota(inds.begin(), inds.end(), 0);
    //sort by coef magnitude
    std::sort(inds.begin(), inds.end(), [&](const size_t &a, const size_t &b) {
        return std::abs(wf.coefs[0][a]) > std::abs(wf.coefs[0][b]);
    });

    for(size_t i : inds) {
        const Det &det = wf.dets[i];
        std::vector<unsigned> up_orbs(det.up.get_occupied_orbs());
        std::vector<unsigned> dn_orbs(det.dn.get_occupied_orbs());

        //print up orbs
        for(unsigned orb : up_orbs) {
            printf("%u ", orb+1);
        }
        //print dn orbs
        printf("\t");
        for(unsigned orb : dn_orbs) {
            printf("%u ", orb+1);
        }
        //print coefs
        for(size_t j = 0; j < wf.coefs.size(); j++) {
            printf("\t% 16.12e", wf.coefs[j][i]);
        }
        printf("\n");
    }
        
    return 0;
}
