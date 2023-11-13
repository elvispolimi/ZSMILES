#pragma once

#include "cuda/node.cuh"

#include <vector>

namespace smiles {
  namespace cuda {
    constexpr std::array<node, 425> build_gpu_smiles_dictionary() {
      std::array<node, 425> n{
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
      n[4].neighbor[34]   = 53;
      n[4].neighbor[37]   = 54;
      n[4].neighbor[46]   = 55;
      n[4].neighbor[45]   = 56;
      n[6].neighbor[8]    = 55;
      n[6].neighbor[34]   = 56;
      n[8].neighbor[7]    = 55;
      n[11].neighbor[8]   = 53;
      n[11].neighbor[34]  = 54;
      n[12].neighbor[8]   = 54;
      n[13].neighbor[8]   = 54;
      n[17].neighbor[34]  = 51;
      n[17].neighbor[77]  = 52;
      n[17].neighbor[45]  = 53;
      n[17].neighbor[46]  = 54;
      n[18].neighbor[34]  = 54;
      n[18].neighbor[46]  = 55;
      n[18].neighbor[45]  = 56;
      n[19].neighbor[8]   = 56;
      n[19].neighbor[34]  = 57;
      n[19].neighbor[28]  = 58;
      n[21].neighbor[81]  = 57;
      n[22].neighbor[8]   = 57;
      n[24].neighbor[34]  = 56;
      n[24].neighbor[15]  = 57;
      n[24].neighbor[8]   = 58;
      n[24].neighbor[7]   = 59;
      n[24].neighbor[16]  = 60;
      n[27].neighbor[34]  = 58;
      n[29].neighbor[15]  = 57;
      n[29].neighbor[16]  = 58;
      n[29].neighbor[77]  = 59;
      n[29].neighbor[66]  = 60;
      n[34].neighbor[15]  = 56;
      n[35].neighbor[28]  = 56;
      n[35].neighbor[34]  = 57;
      n[35].neighbor[37]  = 58;
      n[35].neighbor[46]  = 59;
      n[35].neighbor[45]  = 60;
      n[36].neighbor[15]  = 60;
      n[36].neighbor[45]  = 61;
      n[36].neighbor[16]  = 62;
      n[36].neighbor[7]   = 63;
      n[36].neighbor[46]  = 64;
      n[36].neighbor[34]  = 65;
      n[37].neighbor[34]  = 65;
      n[37].neighbor[8]   = 66;
      n[37].neighbor[7]   = 67;
      n[37].neighbor[28]  = 68;
      n[38].neighbor[7]   = 68;
      n[38].neighbor[15]  = 69;
      n[39].neighbor[28]  = 69;
      n[42].neighbor[34]  = 67;
      n[45].neighbor[8]   = 65;
      n[46].neighbor[45]  = 65;
      n[47].neighbor[16]  = 65;
      n[48].neighbor[34]  = 65;
      n[49].neighbor[66]  = 65;
      n[49].neighbor[77]  = 66;
      n[49].neighbor[8]   = 67;
      n[49].neighbor[78]  = 68;
      n[49].neighbor[82]  = 69;
      n[50].neighbor[15]  = 69;
      n[50].neighbor[7]   = 70;
      n[50].neighbor[16]  = 71;
      n[50].neighbor[66]  = 72;
      n[50].neighbor[17]  = 73;
      n[51].neighbor[66]  = 73;
      n[51].neighbor[77]  = 74;
      n[51].neighbor[8]   = 75;
      n[51].neighbor[17]  = 76;
      n[51].neighbor[78]  = 77;
      n[52].neighbor[77]  = 77;
      n[52].neighbor[34]  = 78;
      n[52].neighbor[45]  = 79;
      n[52].neighbor[46]  = 80;
      n[52].neighbor[37]  = 81;
      n[53].neighbor[15]  = 81;
      n[54].neighbor[66]  = 81;
      n[54].neighbor[8]   = 82;
      n[55].neighbor[66]  = 82;
      n[56].neighbor[66]  = 82;
      n[57].neighbor[8]   = 82;
      n[57].neighbor[34]  = 83;
      n[57].neighbor[75]  = 84;
      n[57].neighbor[15]  = 85;
      n[58].neighbor[8]   = 85;
      n[59].neighbor[8]   = 85;
      n[62].neighbor[34]  = 83;
      n[63].neighbor[28]  = 83;
      n[65].neighbor[34]  = 82;
      n[68].neighbor[31]  = 80;
      n[69].neighbor[39]  = 80;
      n[70].neighbor[10]  = 80;
      n[71].neighbor[12]  = 80;
      n[73].neighbor[8]   = 79;
      n[76].neighbor[34]  = 77;
      n[76].neighbor[7]   = 78;
      n[77].neighbor[34]  = 78;
      n[78].neighbor[8]   = 78;
      n[80].neighbor[7]   = 77;
      n[81].neighbor[34]  = 77;
      n[83].neighbor[34]  = 76;
      n[85].neighbor[28]  = 75;
      n[86].neighbor[8]   = 75;
      n[86].neighbor[77]  = 76;
      n[86].neighbor[66]  = 77;
      n[87].neighbor[8]   = 77;
      n[88].neighbor[15]  = 77;
      n[89].neighbor[15]  = 77;
      n[89].neighbor[7]   = 78;
      n[90].neighbor[8]   = 78;
      n[91].neighbor[46]  = 78;
      n[91].neighbor[34]  = 79;
      n[91].neighbor[50]  = 80;
      n[92].neighbor[8]   = 80;
      n[92].neighbor[34]  = 81;
      n[93].neighbor[8]   = 81;
      n[94].neighbor[8]   = 81;
      n[96].neighbor[8]   = 80;
      n[96].neighbor[34]  = 81;
      n[99].neighbor[34]  = 79;
      n[99].neighbor[28]  = 80;
      n[101].neighbor[15] = 79;
      n[101].neighbor[34] = 80;
      n[102].neighbor[34] = 80;
      n[104].neighbor[34] = 79;
      n[105].neighbor[46] = 79;
      n[106].neighbor[34] = 79;
      n[107].neighbor[34] = 79;
      n[108].neighbor[46] = 79;
      n[111].neighbor[8]  = 77;
      n[112].neighbor[66] = 77;
      n[113].neighbor[31] = 77;
      n[114].neighbor[66] = 77;
      n[114].neighbor[77] = 78;
      n[114].neighbor[82] = 79;
      n[114].neighbor[78] = 80;
      n[115].neighbor[66] = 80;
      n[115].neighbor[78] = 81;
      n[115].neighbor[58] = 82;
      n[115].neighbor[77] = 83;
      n[117].neighbor[77] = 82;
      n[117].neighbor[66] = 83;
      n[119].neighbor[8]  = 82;
      n[122].neighbor[15] = 80;
      n[124].neighbor[66] = 79;
      n[124].neighbor[77] = 80;
      n[127].neighbor[8]  = 78;
      n[129].neighbor[15] = 77;
      n[130].neighbor[8]  = 77;
      n[130].neighbor[75] = 78;
      n[132].neighbor[8]  = 77;
      n[133].neighbor[8]  = 77;
      n[134].neighbor[8]  = 77;
      n[135].neighbor[66] = 77;
      n[137].neighbor[66] = 76;
      n[138].neighbor[66] = 76;
      n[140].neighbor[15] = 75;
      n[141].neighbor[8]  = 75;
      n[142].neighbor[8]  = 75;
      n[146].neighbor[46] = 72;
      n[148].neighbor[31] = 71;
      n[148].neighbor[39] = 72;
      n[148].neighbor[60] = 73;
      n[149].neighbor[60] = 73;
      n[150].neighbor[60] = 73;
      n[151].neighbor[60] = 73;
      n[153].neighbor[16] = 72;
      n[153].neighbor[15] = 73;
      n[154].neighbor[28] = 73;
      n[155].neighbor[7]  = 73;
      n[157].neighbor[28] = 72;
      n[158].neighbor[34] = 72;
      n[159].neighbor[34] = 72;
      n[159].neighbor[8]  = 73;
      n[159].neighbor[7]  = 74;
      n[160].neighbor[34] = 74;
      n[162].neighbor[77] = 73;
      n[163].neighbor[66] = 73;
      n[165].neighbor[8]  = 72;
      n[169].neighbor[8]  = 69;
      n[171].neighbor[8]  = 68;
      n[172].neighbor[7]  = 68;
      n[173].neighbor[15] = 68;
      n[174].neighbor[7]  = 68;
      n[177].neighbor[34] = 66;
      n[178].neighbor[8]  = 66;
      n[179].neighbor[46] = 66;
      n[180].neighbor[8]  = 66;
      n[182].neighbor[15] = 65;
      n[183].neighbor[34] = 65;
      n[184].neighbor[8]  = 65;
      n[185].neighbor[34] = 65;
      n[185].neighbor[15] = 66;
      n[186].neighbor[34] = 66;
      n[189].neighbor[66] = 64;
      n[190].neighbor[39] = 64;
      n[190].neighbor[31] = 65;
      n[191].neighbor[66] = 65;
      n[191].neighbor[7]  = 66;
      n[191].neighbor[77] = 67;
      n[191].neighbor[82] = 68;
      n[192].neighbor[77] = 68;
      n[195].neighbor[7]  = 66;
      n[195].neighbor[66] = 67;
      n[196].neighbor[66] = 67;
      n[197].neighbor[77] = 67;
      n[198].neighbor[77] = 67;
      n[199].neighbor[66] = 67;
      n[200].neighbor[66] = 67;
      n[203].neighbor[66] = 65;
      n[206].neighbor[8]  = 63;
      n[208].neighbor[8]  = 62;
      n[212].neighbor[66] = 59;
      n[213].neighbor[66] = 59;
      n[214].neighbor[66] = 59;
      n[215].neighbor[8]  = 59;
      n[215].neighbor[34] = 60;
      n[218].neighbor[8]  = 58;
      n[219].neighbor[39] = 58;
      n[219].neighbor[60] = 59;
      n[220].neighbor[60] = 59;
      n[223].neighbor[7]  = 57;
      n[224].neighbor[58] = 57;
      n[225].neighbor[66] = 57;
      n[226].neighbor[8]  = 57;
      n[227].neighbor[46] = 57;
      n[229].neighbor[46] = 56;
      n[231].neighbor[15] = 55;
      n[233].neighbor[28] = 54;
      n[234].neighbor[59] = 54;
      n[238].neighbor[45] = 51;
      n[238].neighbor[46] = 52;
      n[238].neighbor[34] = 53;
      n[240].neighbor[34] = 52;
      n[241].neighbor[8]  = 52;
      n[242].neighbor[37] = 52;
      n[244].neighbor[7]  = 51;
      n[245].neighbor[8]  = 51;
      n[247].neighbor[8]  = 50;
      n[251].neighbor[8]  = 47;
      n[254].neighbor[60] = 45;
      n[255].neighbor[39] = 45;
      n[256].neighbor[66] = 45;
      n[256].neighbor[7]  = 46;
      n[256].neighbor[77] = 47;
      n[259].neighbor[66] = 45;
      n[261].neighbor[77] = 44;
      n[263].neighbor[7]  = 43;
      n[264].neighbor[39] = 43;
      n[265].neighbor[77] = 43;
      n[266].neighbor[7]  = 43;
      n[267].neighbor[66] = 43;
      n[268].neighbor[7]  = 43;
      n[268].neighbor[15] = 44;
      n[268].neighbor[66] = 45;
      n[271].neighbor[66] = 43;
      n[272].neighbor[66] = 43;
      n[273].neighbor[66] = 43;
      n[275].neighbor[34] = 42;
      n[276].neighbor[7]  = 42;
      n[277].neighbor[60] = 42;
      n[280].neighbor[28] = 40;
      n[281].neighbor[45] = 40;
      n[282].neighbor[15] = 40;
      n[284].neighbor[8]  = 39;
      n[285].neighbor[8]  = 39;
      n[286].neighbor[8]  = 39;
      n[287].neighbor[46] = 39;
      n[289].neighbor[15] = 38;
      n[290].neighbor[8]  = 38;
      n[290].neighbor[34] = 39;
      n[292].neighbor[8]  = 38;
      n[294].neighbor[8]  = 37;
      n[295].neighbor[34] = 37;
      n[296].neighbor[45] = 37;
      n[300].neighbor[60] = 34;
      n[301].neighbor[66] = 34;
      n[301].neighbor[7]  = 35;
      n[302].neighbor[66] = 35;
      n[302].neighbor[37] = 36;
      n[302].neighbor[34] = 37;
      n[303].neighbor[66] = 37;
      n[304].neighbor[15] = 37;
      n[305].neighbor[58] = 37;
      n[306].neighbor[77] = 37;
      n[307].neighbor[60] = 37;
      n[308].neighbor[15] = 37;
      n[309].neighbor[77] = 37;
      n[310].neighbor[15] = 37;
      n[313].neighbor[7]  = 35;
      n[314].neighbor[66] = 35;
      n[315].neighbor[66] = 35;
      n[316].neighbor[66] = 35;
      n[318].neighbor[28] = 34;
      n[320].neighbor[46] = 33;
      n[321].neighbor[10] = 33;
      n[322].neighbor[66] = 33;
      n[326].neighbor[8]  = 30;
      n[327].neighbor[34] = 30;
      n[329].neighbor[7]  = 29;
      n[332].neighbor[8]  = 27;
      n[335].neighbor[15] = 25;
      n[336].neighbor[66] = 25;
      n[337].neighbor[66] = 25;
      n[338].neighbor[8]  = 25;
      n[339].neighbor[75] = 25;
      n[340].neighbor[15] = 25;
      n[342].neighbor[77] = 24;
      n[343].neighbor[15] = 24;
      n[344].neighbor[66] = 24;
      n[345].neighbor[34] = 24;
      n[346].neighbor[15] = 24;
      n[347].neighbor[8]  = 24;
      n[349].neighbor[16] = 23;
      n[350].neighbor[17] = 23;
      n[351].neighbor[18] = 23;
      n[352].neighbor[46] = 23;
      n[353].neighbor[8]  = 23;
      n[354].neighbor[60] = 23;
      n[355].neighbor[66] = 23;
      n[357].neighbor[34] = 22;
      n[358].neighbor[34] = 22;
      n[359].neighbor[46] = 22;
      n[360].neighbor[8]  = 22;
      n[361].neighbor[15] = 22;
      n[362].neighbor[15] = 22;
      n[363].neighbor[66] = 22;
      n[364].neighbor[8]  = 22;
      n[366].neighbor[39] = 21;
      n[367].neighbor[8]  = 21;
      n[368].neighbor[7]  = 21;
      n[370].neighbor[8]  = 20;
      n[372].neighbor[17] = 19;
      n[373].neighbor[18] = 19;
      n[374].neighbor[19] = 19;
      n[375].neighbor[8]  = 19;
      n[376].neighbor[58] = 19;
      n[377].neighbor[7]  = 19;
      n[378].neighbor[66] = 19;
      n[380].neighbor[8]  = 18;
      n[383].neighbor[8]  = 16;
      n[384].neighbor[8]  = 16;
      n[385].neighbor[66] = 16;
      n[387].neighbor[60] = 15;
      n[389].neighbor[77] = 14;
      n[391].neighbor[8]  = 13;
      n[392].neighbor[8]  = 13;
      n[393].neighbor[8]  = 13;
      n[395].neighbor[46] = 12;
      n[396].neighbor[28] = 12;
      n[397].neighbor[66] = 12;
      n[398].neighbor[7]  = 12;
      n[401].neighbor[15] = 10;
      n[402].neighbor[15] = 10;
      n[403].neighbor[15] = 10;
      n[407].neighbor[12] = 7;
      n[408].neighbor[46] = 7;
      n[409].neighbor[66] = 7;
      n[410].neighbor[34] = 7;
      n[412].neighbor[8]  = 6;
      n[413].neighbor[8]  = 6;
      n[414].neighbor[60] = 6;
      n[415].neighbor[8]  = 6;
      n[416].neighbor[15] = 6;
      n[417].neighbor[8]  = 6;
      n[423].neighbor[34] = 1;
      return n;
    }
  } // namespace cuda
} // namespace smiles