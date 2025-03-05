#pragma once

#include "compression_dictionary.hpp"
#include "gpu/node.hpp"

#include <vector>

#define GPU_DICT_SIZE 429

namespace smiles {
  namespace gpu {
    class smiles_dictionary_entry_gpu {
    public:
      std::string::size_type size;
      char pattern[LONGEST_PATTERN+1];

      constexpr smiles_dictionary_entry_gpu(std::string::size_type size, const char* pattern)
          : size(size), pattern() {
        strcpy(this->pattern, pattern);
      };
      constexpr smiles_dictionary_entry_gpu(): size(0), pattern(){};
    };

    constexpr std::array<smiles_dictionary_entry_gpu, DICT_SIZE> build_gpu_smiles_dictionary_entries() {
      std::array<smiles_dictionary_entry_gpu, DICT_SIZE> n{};
      for (int i = 0; i < DICT_SIZE; i++) {
        n[i] = smiles_dictionary_entry_gpu(SMILES_DICTIONARY[i].size, SMILES_DICTIONARY[i].pattern);
      }
      return n;
    }

    constexpr std::array<node, GPU_DICT_SIZE> build_gpu_smiles_dictionary() {
      std::array<node, GPU_DICT_SIZE> n{
          node{' '},
          node{'C', static_cast<node::node_letter>(67)},
          node{'c', static_cast<node::node_letter>(99)},
          node{'#', static_cast<node::node_letter>(35)},
          node{'(', static_cast<node::node_letter>(40)},
          node{')', static_cast<node::node_letter>(41)},
          node{'0', static_cast<node::node_letter>(48)},
          node{'+', static_cast<node::node_letter>(43)},
          node{'S', static_cast<node::node_letter>(83)},
          node{'-', static_cast<node::node_letter>(45)},
          node{'/', static_cast<node::node_letter>(47)},
          node{'1', static_cast<node::node_letter>(49)},
          node{'2', static_cast<node::node_letter>(50)},
          node{'3', static_cast<node::node_letter>(51)},
          node{'4', static_cast<node::node_letter>(52)},
          node{'5', static_cast<node::node_letter>(53)},
          node{'6', static_cast<node::node_letter>(54)},
          node{'[', static_cast<node::node_letter>(91)},
          node{'=', static_cast<node::node_letter>(61)},
          node{'O', static_cast<node::node_letter>(79)},
          node{'@', static_cast<node::node_letter>(64)},
          node{'B', static_cast<node::node_letter>(66)},
          node{'F', static_cast<node::node_letter>(70)},
          node{'I', static_cast<node::node_letter>(73)},
          node{'N', static_cast<node::node_letter>(78)},
          node{'P', static_cast<node::node_letter>(80)},
          node{'Z', static_cast<node::node_letter>(90)},
          node{'\\', static_cast<node::node_letter>(92)},
          node{']', static_cast<node::node_letter>(93)},
          node{'n', static_cast<node::node_letter>(110)},
          node{'i', static_cast<node::node_letter>(105)},
          node{'l', static_cast<node::node_letter>(108)},
          node{'o', static_cast<node::node_letter>(111)},
          node{'r', static_cast<node::node_letter>(114)},
          node{'s', static_cast<node::node_letter>(115)},
          node{'(', static_cast<node::node_letter>(77)},
          node{'C', static_cast<node::node_letter>(36)},
          node{'0', static_cast<node::node_letter>(69)},
          node{'N', static_cast<node::node_letter>(75)},
          node{'1', static_cast<node::node_letter>(87)},
          node{'O', static_cast<node::node_letter>(89)},
          node{')', static_cast<node::node_letter>(94)},
          node{'='},
          node{'2', static_cast<node::node_letter>(155)},
          node{'S', static_cast<node::node_letter>(169)},
          node{'l', static_cast<node::node_letter>(236)},
          node{'#'},
          node{'c'},
          node{'['},
          node{'0', static_cast<node::node_letter>(37)},
          node{'c', static_cast<node::node_letter>(39)},
          node{'1', static_cast<node::node_letter>(55)},
          node{'(', static_cast<node::node_letter>(57)},
          node{'n', static_cast<node::node_letter>(72)},
          node{'2', static_cast<node::node_letter>(74)},
          node{'3', static_cast<node::node_letter>(151)},
          node{'4'},
          node{'['},
          node{'C', static_cast<node::node_letter>(38)},
          node{'F'},
          node{'O'},
          node{'N', static_cast<node::node_letter>(237)},
          node{')', static_cast<node::node_letter>(42)},
          node{'C', static_cast<node::node_letter>(192)},
          node{'('},
          node{')', static_cast<node::node_letter>(167)},
          node{'C'},
          node{')', static_cast<node::node_letter>(112)},
          node{')', static_cast<node::node_letter>(182)},
          node{'C'},
          node{'n'},
          node{'N'},
          node{'O'},
          node{'C', static_cast<node::node_letter>(102)},
          node{'O'},
          node{'N', static_cast<node::node_letter>(183)},
          node{')', static_cast<node::node_letter>(62)},
          node{'C', static_cast<node::node_letter>(146)},
          node{'='},
          node{'r'},
          node{')', static_cast<node::node_letter>(174)},
          node{'C', static_cast<node::node_letter>(176)},
          node{'0', static_cast<node::node_letter>(251)},
          node{')', static_cast<node::node_letter>(124)},
          node{'('},
          node{'1', static_cast<node::node_letter>(223)},
          node{'C'},
          node{'0', static_cast<node::node_letter>(100)},
          node{'1', static_cast<node::node_letter>(107)},
          node{'n', static_cast<node::node_letter>(116)},
          node{'c'},
          node{'0', static_cast<node::node_letter>(216)},
          node{'='},
          node{'C', static_cast<node::node_letter>(132)},
          node{'F'},
          node{'O'},
          node{'N', static_cast<node::node_letter>(200)},
          node{'0', static_cast<node::node_letter>(96)},
          node{'N', static_cast<node::node_letter>(130)},
          node{'1', static_cast<node::node_letter>(138)},
          node{'(', static_cast<node::node_letter>(148)},
          node{'O', static_cast<node::node_letter>(161)},
          node{'C', static_cast<node::node_letter>(186)},
          node{'C'},
          node{')', static_cast<node::node_letter>(117)},
          node{'('},
          node{'=', static_cast<node::node_letter>(218)},
          node{'('},
          node{'0', static_cast<node::node_letter>(215)},
          node{'='},
          node{'C', static_cast<node::node_letter>(136)},
          node{')', static_cast<node::node_letter>(171)},
          node{'N'},
          node{'1'},
          node{'C'},
          node{'c'},
          node{'n', static_cast<node::node_letter>(81)},
          node{')', static_cast<node::node_letter>(103)},
          node{'o', static_cast<node::node_letter>(131)},
          node{'s', static_cast<node::node_letter>(226)},
          node{'0', static_cast<node::node_letter>(59)},
          node{'(', static_cast<node::node_letter>(125)},
          node{'1', static_cast<node::node_letter>(140)},
          node{'c'},
          node{'2', static_cast<node::node_letter>(217)},
          node{'c'},
          node{'n', static_cast<node::node_letter>(141)},
          node{')', static_cast<node::node_letter>(181)},
          node{'2', static_cast<node::node_letter>(235)},
          node{'o', static_cast<node::node_letter>(245)},
          node{'n'},
          node{'C', static_cast<node::node_letter>(127)},
          node{'N', static_cast<node::node_letter>(230)},
          node{'O'},
          node{'F'},
          node{'0', static_cast<node::node_letter>(234)},
          node{'c'},
          node{')', static_cast<node::node_letter>(225)},
          node{'c'},
          node{'c'},
          node{'n'},
          node{')', static_cast<node::node_letter>(46)},
          node{'C', static_cast<node::node_letter>(65)},
          node{'l'},
          node{'0'},
          node{')', static_cast<node::node_letter>(68)},
          node{')', static_cast<node::node_letter>(119)},
          node{'C', static_cast<node::node_letter>(56)},
          node{'='},
          node{'C', static_cast<node::node_letter>(173)},
          node{'@'},
          node{'H'},
          node{'+'},
          node{'-'},
          node{')', static_cast<node::node_letter>(120)},
          node{'C', static_cast<node::node_letter>(121)},
          node{'('},
          node{'C'},
          node{')', static_cast<node::node_letter>(139)},
          node{'('},
          node{'C'},
          node{'C', static_cast<node::node_letter>(185)},
          node{'='},
          node{')', static_cast<node::node_letter>(180)},
          node{'n'},
          node{'c'},
          node{')', static_cast<node::node_letter>(213)},
          node{'0'},
          node{'0', static_cast<node::node_letter>(134)},
          node{'(', static_cast<node::node_letter>(160)},
          node{')', static_cast<node::node_letter>(137)},
          node{'O'},
          node{'C', static_cast<node::node_letter>(135)},
          node{'S'},
          node{')', static_cast<node::node_letter>(162)},
          node{'C'},
          node{')'},
          node{')', static_cast<node::node_letter>(164)},
          node{')', static_cast<node::node_letter>(106)},
          node{'C'},
          node{'C', static_cast<node::node_letter>(179)},
          node{'='},
          node{'0', static_cast<node::node_letter>(170)},
          node{'C', static_cast<node::node_letter>(203)},
          node{'C', static_cast<node::node_letter>(84)},
          node{'C'},
          node{'O'},
          node{'C', static_cast<node::node_letter>(166)},
          node{'C'},
          node{'O', static_cast<node::node_letter>(190)},
          node{')', static_cast<node::node_letter>(204)},
          node{'c'},
          node{'@'},
          node{'c', static_cast<node::node_letter>(97)},
          node{'n', static_cast<node::node_letter>(118)},
          node{'s', static_cast<node::node_letter>(144)},
          node{'o', static_cast<node::node_letter>(254)},
          node{'c'},
          node{'o'},
          node{'['},
          node{'n'},
          node{'n'},
          node{'c'},
          node{')', static_cast<node::node_letter>(153)},
          node{'0', static_cast<node::node_letter>(154)},
          node{'c', static_cast<node::node_letter>(85)},
          node{'n', static_cast<node::node_letter>(193)},
          node{')', static_cast<node::node_letter>(212)},
          node{'0'},
          node{')', static_cast<node::node_letter>(128)},
          node{'l'},
          node{')', static_cast<node::node_letter>(242)},
          node{')', static_cast<node::node_letter>(244)},
          node{')', static_cast<node::node_letter>(149)},
          node{'c', static_cast<node::node_letter>(145)},
          node{'c'},
          node{'c'},
          node{'H'},
          node{'0', static_cast<node::node_letter>(233)},
          node{')', static_cast<node::node_letter>(104)},
          node{')', static_cast<node::node_letter>(113)},
          node{'O'},
          node{'@'},
          node{'H'},
          node{']', static_cast<node::node_letter>(239)},
          node{']', static_cast<node::node_letter>(60)},
          node{']'},
          node{']'},
          node{'1'},
          node{'0'},
          node{'='},
          node{'(', static_cast<node::node_letter>(238)},
          node{'='},
          node{'C', static_cast<node::node_letter>(109)},
          node{'C'},
          node{')', static_cast<node::node_letter>(191)},
          node{'('},
          node{'C'},
          node{'n', static_cast<node::node_letter>(195)},
          node{'c', static_cast<node::node_letter>(247)},
          node{')', static_cast<node::node_letter>(214)},
          node{')', static_cast<node::node_letter>(33)},
          node{')', static_cast<node::node_letter>(205)},
          node{'('},
          node{'0'},
          node{'('},
          node{'C', static_cast<node::node_letter>(152)},
          node{')', static_cast<node::node_letter>(147)},
          node{'O'},
          node{')', static_cast<node::node_letter>(202)},
          node{'0', static_cast<node::node_letter>(248)},
          node{'C', static_cast<node::node_letter>(178)},
          node{')', static_cast<node::node_letter>(231)},
          node{'C', static_cast<node::node_letter>(184)},
          node{'0'},
          node{'C', static_cast<node::node_letter>(228)},
          node{'c', static_cast<node::node_letter>(208)},
          node{'H'},
          node{'@'},
          node{'c', static_cast<node::node_letter>(34)},
          node{'(', static_cast<node::node_letter>(156)},
          node{'n', static_cast<node::node_letter>(177)},
          node{'s'},
          node{'n', static_cast<node::node_letter>(243)},
          node{'(', static_cast<node::node_letter>(220)},
          node{'c', static_cast<node::node_letter>(165)},
          node{'c'},
          node{'n'},
          node{'n'},
          node{'c'},
          node{'c'},
          node{'c'},
          node{')', static_cast<node::node_letter>(86)},
          node{')', static_cast<node::node_letter>(199)},
          node{'c'},
          node{'c'},
          node{'c'},
          node{']', static_cast<node::node_letter>(255)},
          node{')', static_cast<node::node_letter>(129)},
          node{'C'},
          node{')'},
          node{'H'},
          node{']', static_cast<node::node_letter>(196)},
          node{']', static_cast<node::node_letter>(63)},
          node{'('},
          node{'['},
          node{'c'},
          node{')', static_cast<node::node_letter>(232)},
          node{'O'},
          node{'O'},
          node{'0'},
          node{'='},
          node{'\\', static_cast<node::node_letter>(198)},
          node{'N', static_cast<node::node_letter>(71)},
          node{'O', static_cast<node::node_letter>(133)},
          node{'C', static_cast<node::node_letter>(201)},
          node{'C'},
          node{')', static_cast<node::node_letter>(250)},
          node{'F'},
          node{'('},
          node{')'},
          node{')', static_cast<node::node_letter>(219)},
          node{')', static_cast<node::node_letter>(207)},
          node{']', static_cast<node::node_letter>(227)},
          node{'H'},
          node{'c'},
          node{'(', static_cast<node::node_letter>(150)},
          node{'n'},
          node{'c'},
          node{'n'},
          node{'('},
          node{'H'},
          node{'n'},
          node{'('},
          node{'c'},
          node{'(', static_cast<node::node_letter>(158)},
          node{'0', static_cast<node::node_letter>(240)},
          node{'c'},
          node{'c'},
          node{'c'},
          node{'c'},
          node{'C', static_cast<node::node_letter>(221)},
          node{'('},
          node{']', static_cast<node::node_letter>(58)},
          node{'='},
          node{'N'},
          node{'0'},
          node{')', static_cast<node::node_letter>(211)},
          node{')', static_cast<node::node_letter>(98)},
          node{')', static_cast<node::node_letter>(189)},
          node{'O'},
          node{'0'},
          node{')', static_cast<node::node_letter>(157)},
          node{'C'},
          node{')', static_cast<node::node_letter>(88)},
          node{')', static_cast<node::node_letter>(126)},
          node{'C'},
          node{'N', static_cast<node::node_letter>(252)},
          node{']', static_cast<node::node_letter>(229)},
          node{'c'},
          node{'('},
          node{'c'},
          node{'F'},
          node{'C'},
          node{'c'},
          node{'0', static_cast<node::node_letter>(206)},
          node{'['},
          node{'n'},
          node{']'},
          node{'0'},
          node{'n'},
          node{'0'},
          node{'(', static_cast<node::node_letter>(246)},
          node{'c'},
          node{'c'},
          node{'c'},
          node{'='},
          node{'O'},
          node{'+'},
          node{'c'},
          node{')', static_cast<node::node_letter>(241)},
          node{'C'},
          node{'('},
          node{')'},
          node{'0', static_cast<node::node_letter>(76)},
          node{'c'},
          node{'c'},
          node{')'},
          node{'l'},
          node{'0', static_cast<node::node_letter>(209)},
          node{'n'},
          node{'0'},
          node{'c'},
          node{'C', static_cast<node::node_letter>(210)},
          node{'0'},
          node{')', static_cast<node::node_letter>(222)},
          node{'1'},
          node{'2'},
          node{'3'},
          node{'O'},
          node{')'},
          node{']'},
          node{'c'},
          node{'C', static_cast<node::node_letter>(159)},
          node{'C'},
          node{'O', static_cast<node::node_letter>(172)},
          node{')', static_cast<node::node_letter>(123)},
          node{'0'},
          node{'0'},
          node{'c'},
          node{')', static_cast<node::node_letter>(249)},
          node{'H'},
          node{')', static_cast<node::node_letter>(168)},
          node{'('},
          node{')', static_cast<node::node_letter>(163)},
          node{'2', static_cast<node::node_letter>(101)},
          node{'3'},
          node{'4'},
          node{')', static_cast<node::node_letter>(44)},
          node{'['},
          node{'('},
          node{'c'},
          node{')'},
          node{')', static_cast<node::node_letter>(143)},
          node{')', static_cast<node::node_letter>(82)},
          node{'c'},
          node{']'},
          node{'n'},
          node{')', static_cast<node::node_letter>(253)},
          node{')', static_cast<node::node_letter>(122)},
          node{')', static_cast<node::node_letter>(197)},
          node{'O'},
          node{'='},
          node{'c'},
          node{'('},
          node{'0', static_cast<node::node_letter>(224)},
          node{'0'},
          node{'0'},
          node{'-'},
          node{'O'},
          node{'c'},
          node{'C'},
          node{')', static_cast<node::node_letter>(142)},
          node{')', static_cast<node::node_letter>(188)},
          node{']', static_cast<node::node_letter>(95)},
          node{')', static_cast<node::node_letter>(187)},
          node{'0', static_cast<node::node_letter>(194)},
          node{')'},
          node{'C', static_cast<node::node_letter>(175)},
      };
      n[0].neighbor[34]   = 1;
      n[0].neighbor[66]   = 2;
      n[0].neighbor[2]    = 3;
      n[0].neighbor[7]    = 4;
      n[0].neighbor[8]    = 5;
      n[0].neighbor[15]   = 6;
      n[0].neighbor[10]   = 7;
      n[0].neighbor[50]   = 8;
      n[0].neighbor[12]   = 9;
      n[0].neighbor[14]   = 10;
      n[0].neighbor[16]   = 11;
      n[0].neighbor[17]   = 12;
      n[0].neighbor[18]   = 13;
      n[0].neighbor[19]   = 14;
      n[0].neighbor[20]   = 15;
      n[0].neighbor[21]   = 16;
      n[0].neighbor[58]   = 17;
      n[0].neighbor[28]   = 18;
      n[0].neighbor[46]   = 19;
      n[0].neighbor[31]   = 20;
      n[0].neighbor[33]   = 21;
      n[0].neighbor[37]   = 22;
      n[0].neighbor[40]   = 23;
      n[0].neighbor[45]   = 24;
      n[0].neighbor[47]   = 25;
      n[0].neighbor[57]   = 26;
      n[0].neighbor[59]   = 27;
      n[0].neighbor[60]   = 28;
      n[0].neighbor[77]   = 29;
      n[0].neighbor[72]   = 30;
      n[0].neighbor[75]   = 31;
      n[0].neighbor[78]   = 32;
      n[0].neighbor[81]   = 33;
      n[0].neighbor[82]   = 34;
      n[1].neighbor[7]    = 34;
      n[1].neighbor[34]   = 35;
      n[1].neighbor[15]   = 36;
      n[1].neighbor[45]   = 37;
      n[1].neighbor[16]   = 38;
      n[1].neighbor[46]   = 39;
      n[1].neighbor[8]    = 40;
      n[1].neighbor[28]   = 41;
      n[1].neighbor[17]   = 42;
      n[1].neighbor[50]   = 43;
      n[1].neighbor[75]   = 44;
      n[1].neighbor[2]    = 45;
      n[1].neighbor[66]   = 46;
      n[1].neighbor[58]   = 47;
      n[2].neighbor[15]   = 47;
      n[2].neighbor[66]   = 48;
      n[2].neighbor[16]   = 49;
      n[2].neighbor[7]    = 50;
      n[2].neighbor[77]   = 51;
      n[2].neighbor[17]   = 52;
      n[2].neighbor[18]   = 53;
      n[2].neighbor[19]   = 54;
      n[2].neighbor[58]   = 55;
      n[4].neighbor[34]   = 54;
      n[4].neighbor[37]   = 55;
      n[4].neighbor[46]   = 56;
      n[4].neighbor[45]   = 57;
      n[6].neighbor[8]    = 56;
      n[6].neighbor[34]   = 57;
      n[8].neighbor[7]    = 56;
      n[11].neighbor[8]   = 54;
      n[11].neighbor[34]  = 55;
      n[12].neighbor[8]   = 55;
      n[13].neighbor[8]   = 55;
      n[17].neighbor[34]  = 52;
      n[17].neighbor[77]  = 53;
      n[17].neighbor[45]  = 54;
      n[17].neighbor[46]  = 55;
      n[18].neighbor[34]  = 55;
      n[18].neighbor[46]  = 56;
      n[18].neighbor[45]  = 57;
      n[19].neighbor[8]   = 57;
      n[19].neighbor[34]  = 58;
      n[19].neighbor[28]  = 59;
      n[21].neighbor[81]  = 58;
      n[22].neighbor[8]   = 58;
      n[24].neighbor[34]  = 57;
      n[24].neighbor[15]  = 58;
      n[24].neighbor[8]   = 59;
      n[24].neighbor[7]   = 60;
      n[24].neighbor[16]  = 61;
      n[27].neighbor[34]  = 59;
      n[29].neighbor[15]  = 58;
      n[29].neighbor[16]  = 59;
      n[29].neighbor[77]  = 60;
      n[29].neighbor[66]  = 61;
      n[34].neighbor[15]  = 57;
      n[35].neighbor[28]  = 57;
      n[35].neighbor[34]  = 58;
      n[35].neighbor[37]  = 59;
      n[35].neighbor[46]  = 60;
      n[35].neighbor[45]  = 61;
      n[36].neighbor[15]  = 61;
      n[36].neighbor[45]  = 62;
      n[36].neighbor[16]  = 63;
      n[36].neighbor[7]   = 64;
      n[36].neighbor[46]  = 65;
      n[36].neighbor[34]  = 66;
      n[37].neighbor[34]  = 66;
      n[37].neighbor[8]   = 67;
      n[37].neighbor[7]   = 68;
      n[37].neighbor[28]  = 69;
      n[38].neighbor[7]   = 69;
      n[38].neighbor[15]  = 70;
      n[39].neighbor[28]  = 70;
      n[42].neighbor[34]  = 68;
      n[45].neighbor[8]   = 66;
      n[46].neighbor[45]  = 66;
      n[47].neighbor[16]  = 66;
      n[48].neighbor[34]  = 66;
      n[49].neighbor[66]  = 66;
      n[49].neighbor[77]  = 67;
      n[49].neighbor[8]   = 68;
      n[49].neighbor[78]  = 69;
      n[49].neighbor[82]  = 70;
      n[50].neighbor[15]  = 70;
      n[50].neighbor[7]   = 71;
      n[50].neighbor[16]  = 72;
      n[50].neighbor[66]  = 73;
      n[50].neighbor[17]  = 74;
      n[51].neighbor[66]  = 74;
      n[51].neighbor[77]  = 75;
      n[51].neighbor[8]   = 76;
      n[51].neighbor[17]  = 77;
      n[51].neighbor[78]  = 78;
      n[52].neighbor[77]  = 78;
      n[52].neighbor[34]  = 79;
      n[52].neighbor[45]  = 80;
      n[52].neighbor[46]  = 81;
      n[52].neighbor[37]  = 82;
      n[53].neighbor[15]  = 82;
      n[54].neighbor[66]  = 82;
      n[54].neighbor[8]   = 83;
      n[55].neighbor[66]  = 83;
      n[56].neighbor[66]  = 83;
      n[57].neighbor[77]  = 83;
      n[58].neighbor[8]   = 83;
      n[58].neighbor[34]  = 84;
      n[58].neighbor[75]  = 85;
      n[58].neighbor[15]  = 86;
      n[59].neighbor[8]   = 86;
      n[60].neighbor[8]   = 86;
      n[63].neighbor[34]  = 84;
      n[64].neighbor[28]  = 84;
      n[66].neighbor[34]  = 83;
      n[69].neighbor[31]  = 81;
      n[70].neighbor[39]  = 81;
      n[71].neighbor[10]  = 81;
      n[72].neighbor[12]  = 81;
      n[74].neighbor[8]   = 80;
      n[77].neighbor[34]  = 78;
      n[77].neighbor[7]   = 79;
      n[78].neighbor[34]  = 79;
      n[79].neighbor[8]   = 79;
      n[81].neighbor[7]   = 78;
      n[82].neighbor[34]  = 78;
      n[84].neighbor[34]  = 77;
      n[86].neighbor[28]  = 76;
      n[87].neighbor[8]   = 76;
      n[87].neighbor[77]  = 77;
      n[87].neighbor[66]  = 78;
      n[88].neighbor[8]   = 78;
      n[89].neighbor[15]  = 78;
      n[90].neighbor[15]  = 78;
      n[90].neighbor[7]   = 79;
      n[91].neighbor[8]   = 79;
      n[92].neighbor[46]  = 79;
      n[92].neighbor[34]  = 80;
      n[92].neighbor[50]  = 81;
      n[93].neighbor[8]   = 81;
      n[93].neighbor[34]  = 82;
      n[94].neighbor[8]   = 82;
      n[95].neighbor[8]   = 82;
      n[97].neighbor[8]   = 81;
      n[97].neighbor[34]  = 82;
      n[100].neighbor[34] = 80;
      n[100].neighbor[28] = 81;
      n[102].neighbor[15] = 80;
      n[102].neighbor[34] = 81;
      n[103].neighbor[34] = 81;
      n[105].neighbor[34] = 80;
      n[106].neighbor[46] = 80;
      n[107].neighbor[34] = 80;
      n[108].neighbor[34] = 80;
      n[109].neighbor[46] = 80;
      n[112].neighbor[8]  = 78;
      n[113].neighbor[66] = 78;
      n[114].neighbor[31] = 78;
      n[115].neighbor[66] = 78;
      n[115].neighbor[77] = 79;
      n[115].neighbor[82] = 80;
      n[115].neighbor[78] = 81;
      n[116].neighbor[66] = 81;
      n[116].neighbor[78] = 82;
      n[116].neighbor[58] = 83;
      n[116].neighbor[77] = 84;
      n[118].neighbor[77] = 83;
      n[118].neighbor[66] = 84;
      n[120].neighbor[8]  = 83;
      n[123].neighbor[15] = 81;
      n[125].neighbor[66] = 80;
      n[125].neighbor[77] = 81;
      n[128].neighbor[8]  = 79;
      n[130].neighbor[15] = 78;
      n[131].neighbor[8]  = 78;
      n[131].neighbor[75] = 79;
      n[133].neighbor[8]  = 78;
      n[134].neighbor[8]  = 78;
      n[135].neighbor[8]  = 78;
      n[136].neighbor[66] = 78;
      n[138].neighbor[66] = 77;
      n[139].neighbor[66] = 77;
      n[140].neighbor[39] = 77;
      n[142].neighbor[15] = 76;
      n[143].neighbor[8]  = 76;
      n[144].neighbor[8]  = 76;
      n[148].neighbor[46] = 73;
      n[150].neighbor[31] = 72;
      n[150].neighbor[39] = 73;
      n[150].neighbor[60] = 74;
      n[151].neighbor[60] = 74;
      n[152].neighbor[60] = 74;
      n[153].neighbor[60] = 74;
      n[155].neighbor[16] = 73;
      n[155].neighbor[15] = 74;
      n[156].neighbor[28] = 74;
      n[157].neighbor[7]  = 74;
      n[159].neighbor[28] = 73;
      n[160].neighbor[34] = 73;
      n[161].neighbor[34] = 73;
      n[161].neighbor[8]  = 74;
      n[161].neighbor[7]  = 75;
      n[162].neighbor[34] = 75;
      n[164].neighbor[77] = 74;
      n[165].neighbor[66] = 74;
      n[167].neighbor[8]  = 73;
      n[171].neighbor[8]  = 70;
      n[173].neighbor[8]  = 69;
      n[174].neighbor[7]  = 69;
      n[175].neighbor[15] = 69;
      n[176].neighbor[7]  = 69;
      n[179].neighbor[34] = 67;
      n[180].neighbor[8]  = 67;
      n[181].neighbor[46] = 67;
      n[182].neighbor[8]  = 67;
      n[184].neighbor[15] = 66;
      n[185].neighbor[34] = 66;
      n[186].neighbor[8]  = 66;
      n[187].neighbor[34] = 66;
      n[187].neighbor[15] = 67;
      n[188].neighbor[34] = 67;
      n[191].neighbor[66] = 65;
      n[192].neighbor[39] = 65;
      n[192].neighbor[31] = 66;
      n[193].neighbor[66] = 66;
      n[193].neighbor[7]  = 67;
      n[193].neighbor[77] = 68;
      n[193].neighbor[82] = 69;
      n[194].neighbor[77] = 69;
      n[197].neighbor[7]  = 67;
      n[197].neighbor[66] = 68;
      n[198].neighbor[66] = 68;
      n[199].neighbor[77] = 68;
      n[200].neighbor[77] = 68;
      n[201].neighbor[66] = 68;
      n[202].neighbor[66] = 68;
      n[205].neighbor[66] = 66;
      n[208].neighbor[8]  = 64;
      n[210].neighbor[8]  = 63;
      n[214].neighbor[66] = 60;
      n[215].neighbor[66] = 60;
      n[216].neighbor[66] = 60;
      n[217].neighbor[60] = 60;
      n[218].neighbor[8]  = 60;
      n[218].neighbor[34] = 61;
      n[221].neighbor[8]  = 59;
      n[222].neighbor[39] = 59;
      n[222].neighbor[60] = 60;
      n[223].neighbor[60] = 60;
      n[226].neighbor[7]  = 58;
      n[227].neighbor[58] = 58;
      n[228].neighbor[66] = 58;
      n[229].neighbor[8]  = 58;
      n[230].neighbor[46] = 58;
      n[232].neighbor[46] = 57;
      n[234].neighbor[15] = 56;
      n[236].neighbor[28] = 55;
      n[237].neighbor[59] = 55;
      n[241].neighbor[45] = 52;
      n[241].neighbor[46] = 53;
      n[241].neighbor[34] = 54;
      n[243].neighbor[34] = 53;
      n[244].neighbor[8]  = 53;
      n[245].neighbor[37] = 53;
      n[247].neighbor[7]  = 52;
      n[248].neighbor[8]  = 52;
      n[250].neighbor[8]  = 51;
      n[254].neighbor[8]  = 48;
      n[257].neighbor[60] = 46;
      n[258].neighbor[39] = 46;
      n[259].neighbor[66] = 46;
      n[259].neighbor[7]  = 47;
      n[259].neighbor[77] = 48;
      n[262].neighbor[66] = 46;
      n[264].neighbor[77] = 45;
      n[266].neighbor[7]  = 44;
      n[267].neighbor[39] = 44;
      n[268].neighbor[77] = 44;
      n[269].neighbor[7]  = 44;
      n[270].neighbor[66] = 44;
      n[271].neighbor[7]  = 44;
      n[271].neighbor[15] = 45;
      n[271].neighbor[66] = 46;
      n[274].neighbor[66] = 44;
      n[275].neighbor[66] = 44;
      n[276].neighbor[66] = 44;
      n[279].neighbor[34] = 42;
      n[280].neighbor[7]  = 42;
      n[281].neighbor[60] = 42;
      n[284].neighbor[28] = 40;
      n[285].neighbor[45] = 40;
      n[286].neighbor[15] = 40;
      n[288].neighbor[8]  = 39;
      n[289].neighbor[8]  = 39;
      n[290].neighbor[8]  = 39;
      n[291].neighbor[46] = 39;
      n[293].neighbor[15] = 38;
      n[294].neighbor[8]  = 38;
      n[294].neighbor[34] = 39;
      n[296].neighbor[8]  = 38;
      n[298].neighbor[8]  = 37;
      n[299].neighbor[34] = 37;
      n[300].neighbor[45] = 37;
      n[304].neighbor[60] = 34;
      n[305].neighbor[66] = 34;
      n[305].neighbor[7]  = 35;
      n[306].neighbor[66] = 35;
      n[306].neighbor[37] = 36;
      n[306].neighbor[34] = 37;
      n[307].neighbor[66] = 37;
      n[308].neighbor[15] = 37;
      n[309].neighbor[58] = 37;
      n[310].neighbor[77] = 37;
      n[311].neighbor[60] = 37;
      n[312].neighbor[15] = 37;
      n[313].neighbor[77] = 37;
      n[314].neighbor[15] = 37;
      n[317].neighbor[7]  = 35;
      n[318].neighbor[66] = 35;
      n[319].neighbor[66] = 35;
      n[320].neighbor[66] = 35;
      n[322].neighbor[28] = 34;
      n[324].neighbor[46] = 33;
      n[325].neighbor[10] = 33;
      n[326].neighbor[66] = 33;
      n[330].neighbor[8]  = 30;
      n[331].neighbor[34] = 30;
      n[333].neighbor[7]  = 29;
      n[336].neighbor[8]  = 27;
      n[339].neighbor[15] = 25;
      n[340].neighbor[66] = 25;
      n[341].neighbor[66] = 25;
      n[342].neighbor[8]  = 25;
      n[343].neighbor[75] = 25;
      n[344].neighbor[15] = 25;
      n[346].neighbor[77] = 24;
      n[347].neighbor[15] = 24;
      n[348].neighbor[66] = 24;
      n[349].neighbor[34] = 24;
      n[350].neighbor[15] = 24;
      n[351].neighbor[8]  = 24;
      n[353].neighbor[16] = 23;
      n[354].neighbor[17] = 23;
      n[355].neighbor[18] = 23;
      n[356].neighbor[46] = 23;
      n[357].neighbor[8]  = 23;
      n[358].neighbor[60] = 23;
      n[359].neighbor[66] = 23;
      n[361].neighbor[34] = 22;
      n[362].neighbor[34] = 22;
      n[363].neighbor[46] = 22;
      n[364].neighbor[8]  = 22;
      n[365].neighbor[15] = 22;
      n[366].neighbor[15] = 22;
      n[367].neighbor[66] = 22;
      n[368].neighbor[8]  = 22;
      n[370].neighbor[39] = 21;
      n[371].neighbor[8]  = 21;
      n[372].neighbor[7]  = 21;
      n[374].neighbor[8]  = 20;
      n[376].neighbor[17] = 19;
      n[377].neighbor[18] = 19;
      n[378].neighbor[19] = 19;
      n[379].neighbor[8]  = 19;
      n[380].neighbor[58] = 19;
      n[381].neighbor[7]  = 19;
      n[382].neighbor[66] = 19;
      n[384].neighbor[8]  = 18;
      n[387].neighbor[8]  = 16;
      n[388].neighbor[8]  = 16;
      n[389].neighbor[66] = 16;
      n[391].neighbor[60] = 15;
      n[393].neighbor[77] = 14;
      n[395].neighbor[8]  = 13;
      n[396].neighbor[8]  = 13;
      n[397].neighbor[8]  = 13;
      n[399].neighbor[46] = 12;
      n[400].neighbor[28] = 12;
      n[401].neighbor[66] = 12;
      n[402].neighbor[7]  = 12;
      n[405].neighbor[15] = 10;
      n[406].neighbor[15] = 10;
      n[407].neighbor[15] = 10;
      n[411].neighbor[12] = 7;
      n[412].neighbor[46] = 7;
      n[413].neighbor[66] = 7;
      n[414].neighbor[34] = 7;
      n[416].neighbor[8]  = 6;
      n[417].neighbor[8]  = 6;
      n[418].neighbor[60] = 6;
      n[419].neighbor[8]  = 6;
      n[420].neighbor[15] = 6;
      n[421].neighbor[8]  = 6;
      n[427].neighbor[34] = 1;
      return n;
    }
  } // namespace gpu
} // namespace smiles