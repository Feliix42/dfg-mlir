/*
func.func private @source() -> (i64, i64, i64)
func.func private @sum(i64, i64) -> i64
func.func private @mul(i64, i64) -> i64
func.func private @sink(i64) -> ()
*/

#include <stdint.h>

#include <iostream>
#include <array>
#include <random>

#include <openssl/sha.h>

#define D_PBLC 0x8080
#define D_MESG 0x8181
#define D_LEAF 0x8282
#define D_INTR 0x8383

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;

/*LMS*/


// a second-preimage-resistant cryptographic hash function that accepts byte
// strings of any length and returns an n-byte string. SHA256 in our case
template <size_t size>
std::array<u8, 32> H(std::array<u8, size> message) {
  std::array<u8, 32> retVal;
  SHA256(message.data(), message.size(), retVal.data());
  return retVal;
}

template <size_t size1, size_t size2>
std::array<u8, size1 + size2> operator||(std::array<u8, size1> array1,
                                         std::array<u8, size2> array2) {
  std::array<u8, size1 + size2> retVal;
  for (int i = 0; i < array1.size(); i++) {
    retVal[i] = array1[i];
  }
  for (int i = 0; i < array2.size(); i++) {
    retVal[i + array1.size()] = array2[i];
  }
  return retVal;
}


// If S is a string, i is a positive integer, and w is a member of the set { 1,
// 2, 4, 8 }, then coef(S, i, w) is the i-th, w-bit value, if S is interpreted
// as a sequence of w-bit values.
template <size_t size>
u8 coef(std::array<u8, size> S, u8 i, u8 w) {
  return ((1 << w) - 1) & (S[i * w / 8] >> (8 - (w * (i % (8 / w)) + w)));
}

template <size_t size>
u16 Cksm(std::array<u8, size> S, u32 n, u32 w, u32 ls) {
  u16 sum = 0;
  for (int i = 0; i < (n * 8 / w); i++) {
    sum = sum + (2 ^ w - 1) - coef(S, i, w);
  }
  return (sum << ls);
}

std::array<u8, 4> u32str(int x) {
  std::array<u8, 4> retVal;
  for (u8 i = 0, shift = 24; i < 4; i++, shift -= 8) {
    retVal[i] = x >> shift;
  }
  return retVal;
}

std::array<u8, 2> u16str(int x) {
  std::array<u8, 2> retVal;
  for (u8 i = 0, shift = 8; i < 2; i++, shift -= 8) {
    retVal[i] = x >> shift;
  }
  return retVal;
}

std::array<u8, 1> u8str(int x) { return {(u8)(x & 0xFF)}; }

std::random_device rd;
std::mt19937 gen(rd());  // Mersenne Twister engine

// Define a uniform distribution for bytes (0 to 255)
std::uniform_int_distribution<unsigned int> byteDistribution(0, 255);

enum typecode {
  LMOTS_SHA256_N32_W1,
  LMOTS_SHA256_N32_W2,
  LMOTS_SHA256_N32_W4,
  LMOTS_SHA256_N32_W8
};

const typecode chosen = LMOTS_SHA256_N32_W4;
const u32 msg_len = 5;
const u32 signature_amount = 150000;

template <typecode type>
class lmots_private_key {
 public:
  lmots_private_key(u32 _q, std::array<u8, 16> _I) : q(_q), I(_I) {
    for (u32 i = 0; i < p; i++) {
      for (u32 j = 0; j < n; j++) {
        x[i][j] = byteDistribution(gen);
      }
    }
  }
  static constexpr std::array<u32, 4> params() {
    switch (type) {
      case LMOTS_SHA256_N32_W1:
        return {32, 265, 1, 7};
      case LMOTS_SHA256_N32_W2:
        return {32, 133, 2, 6};
      case LMOTS_SHA256_N32_W4:
        return {32, 67, 4, 4};
      case LMOTS_SHA256_N32_W8:
        return {32, 34, 8, 0};
    }
  }

 public:
  static constexpr std::array<u32, 4> data = params();
  static constexpr u32 n = data[0];
  static constexpr u32 p = data[1];
  static constexpr u32 w = data[2];
  static constexpr u32 ls = data[3];
  u32 q;
  std::array<u8, 16> I;
  std::array<std::array<u8, n>, p> x;
};

template <typecode type>
class lmots_public_key {
 public:
  lmots_public_key(lmots_private_key<type> prk) {
    K = H(prk.I || u32str(prk.q) || u16str(D_PBLC));
    for (int i = 0; i < p; i = i + 1) {
      std::array<u8, n> tmp = prk.x[i];
      for (int j = 0; j < (1 << w) - 1; j = j + 1) {
        tmp = H(prk.I || u32str(prk.q) || u16str(i) || u8str(j) || tmp);
      }
      K = H(K || tmp);
    }
    q = prk.q;
    I = prk.I;
  }

  static constexpr std::array<u32, 4> params() {
    switch (type) {
      case LMOTS_SHA256_N32_W1:
        return {32, 265, 1, 7};
      case LMOTS_SHA256_N32_W2:
        return {32, 133, 2, 6};
      case LMOTS_SHA256_N32_W4:
        return {32, 67, 4, 4};
      case LMOTS_SHA256_N32_W8:
        return {32, 34, 8, 0};
    }
  }

 public:
  static constexpr std::array<u32, 4> data = params();
  static constexpr u32 n = data[0];
  static constexpr u32 p = data[1];
  static constexpr u32 w = data[2];
  static constexpr u32 ls = data[3];
  u32 q;
  std::array<u8, 16> I;
  std::array<u8, n> K;
};

template <typecode type>
class lmots_signature {
 public:
  static constexpr std::array<u32, 2> params() {
    switch (type) {
      case LMOTS_SHA256_N32_W1:
        return {32, 265};
      case LMOTS_SHA256_N32_W2:
        return {32, 133};
      case LMOTS_SHA256_N32_W4:
        return {32, 67};
      case LMOTS_SHA256_N32_W8:
        return {32, 34};
    }
  }

 public:
  static constexpr std::array<u32, 2> data = params();
  static constexpr u32 n = data[0];
  static constexpr u32 p = data[1];

  std::array<u8, n> C;
  std::array<std::array<u8, n>, p> y;
};




bool keys_ready = 0;
lmots_private_key<chosen> *privkey;
lmots_public_key<chosen> *pubkey;

extern "C" {

int64_t mul(int64_t in, int64_t out){
	return in*out;
}

void sink(int64_t in){
	std::cout << "Sink found item: " << in << std::endl;
	return;
}


void lms_fill_message(int32_t data, std::array<uint8_t, 5> *empty_data) {
	for(int i = 0 ; i < 5 ; i++){
		(*empty_data)[i] = 'K' + i;
	}
	if(!keys_ready) {
		std::cout << "Keys are not ready, generating keys\n";
		std::array<u8, 16> I;
		for (int i = 0; i < 16; i++) {
			I[i] = byteDistribution(gen);
		}
		// q can be 0 because we only send 1 msg
		privkey = new lmots_private_key<chosen>{0, I};
		pubkey = new lmots_public_key<chosen>{*privkey};
		keys_ready = 1;
	}
}

void lms_sink(int32_t valid){
	if(valid){
		std::cout << "Received a valid sig" << std::endl;
	}else{
		std::cout << "ERROR: Received invalid sig" << std::endl;
	}
}

int32_t lms_get_iterations() {
	return 5;
}


void lms_sign(std::array<uint8_t, 5> *msg_ptr, lmots_signature<chosen> *empty_signature) {

	std::array<uint8_t, 5> msg = *msg_ptr;
	lmots_private_key<chosen> priv_key = *privkey;

	// Some salt
	std::array<u8, priv_key.n> C;
	for (int i = 0; i < priv_key.n; i++) {
		C[i] = byteDistribution(gen);
	}
	std::array<std::array<u8, priv_key.n>, priv_key.p> y;
	std::array<u8, priv_key.n> Q = H(priv_key.I || u32str(priv_key.q) || u16str(D_MESG) || C || msg);
	for (int i = 0; i < priv_key.p; i = i + 1) {
		u8 a = coef(Q || u16str(Cksm(Q, priv_key.n, priv_key.w, priv_key.ls)), i, priv_key.w);
		std::array<u8, priv_key.n> tmp = priv_key.x[i];
		for (int j = 0; j < a; j = j + 1) {
			tmp = H(priv_key.I || u32str(priv_key.q) || u16str(i) || u8str(j) || tmp);
		}
		y[i] = tmp;
	}

	*empty_signature = {C, y};
}


int32_t lms_verify(std::array<uint8_t, 5> *msg_ptr, lmots_signature<chosen> *signature){
	// Verifying the signature

	lmots_public_key<chosen> pub_key = *pubkey;
	std::array<uint8_t, 5> msg = *msg_ptr;

	lmots_signature<chosen> sig = *signature;

	std::array<u8, pub_key.n> Kc = H(pub_key.I || u32str(pub_key.q) || u16str(D_PBLC));
	std::array<u8, pub_key.n> Q = H(pub_key.I || u32str(pub_key.q) || u16str(D_MESG) || sig.C || msg);
	for (int i = 0; i < pub_key.p; i = i + 1) {
		u8 a = coef(Q || u16str(Cksm(Q, pub_key.n, pub_key.w, pub_key.ls)), i, pub_key.w);
		std::array<u8, pub_key.n> tmp = sig.y[i];
		for (int j = a; j < (1 << pub_key.w) - 1; j = j + 1) {
			tmp = H(pub_key.I || u32str(pub_key.q) || u16str(i) || u8str(j) || tmp);
		}
		Kc = H(Kc || tmp);
	}

	for (int i = 0; i < pub_key.n; i++) {
		if (pub_key.K[i] != Kc[i]) {
			return 0;
		}
	}
	return 1;
}

}
