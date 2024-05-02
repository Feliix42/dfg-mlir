std::tuple<int64_t, int64_t, int64_t> source() {
}

int64_t sum() {
}

int64_t mul() {
}

void sink() {
}

void source_wrap(Application* v1, InputRTChannels<> v2, OutputRTChannels<> v3) {
  Channel v4;
  Channel v5;
  Channel v6;
  int64_t v7;
  int64_t v8;
  int64_t v9;
  Channel v10;
  Channel v11;
  Channel v12;
  v4 = std::get<0>(v3);
  v5 = std::get<1>(v3);
  v6 = std::get<2>(v3);
  v10 = v4;
  v11 = v5;
  v12 = v6;
  goto label2;
label2:
  std::tie(v7, v8, v9) = source();
  v3->Push(v7);
  v4->Push(v8);
  v5->Push(v9);
  return;
}

void sum_wrap(Application* v1, InputRTChannels<> v2, OutputRTChannels<> v3) {
  Channel v4;
  Channel v5;
  Channel v6;
  int64_t v7;
  int64_t v8;
  int64_t v9;
  Channel v10;
  Channel v11;
  Channel v12;
  v4 = std::get<0>(v2);
  v5 = std::get<1>(v2);
  v6 = std::get<0>(v3);
  v10 = v4;
  v11 = v5;
  v12 = v6;
  goto label2;
label2:
  v7 = Pop(v10);
  v8 = Pop(v11);
  v9 = sum(v7, v8);
  v5->Push(v9);
  return;
}

void mul_wrap(Application* v1, InputRTChannels<> v2, OutputRTChannels<> v3) {
  Channel v4;
  Channel v5;
  Channel v6;
  int64_t v7;
  int64_t v8;
  int64_t v9;
  Channel v10;
  Channel v11;
  Channel v12;
  v4 = std::get<0>(v2);
  v5 = std::get<1>(v2);
  v6 = std::get<0>(v3);
  v10 = v4;
  v11 = v5;
  v12 = v6;
  goto label2;
label2:
  v7 = Pop(v10);
  v8 = Pop(v11);
  v9 = mul(v7, v8);
  v5->Push(v9);
  return;
}

void sink_wrap(Application* v1, InputRTChannels<> v2, OutputRTChannels<> v3) {
  Channel v4;
  int64_t v5;
  Channel v6;
  v4 = std::get<0>(v2);
  v6 = v4;
  goto label2;
label2:
  v5 = Pop(v6);
  sink(v5);
  return;
}

void main() {
  Channel v1;
  Channel v2;
  Channel v3;
  Channel v4;
  Channel v5;
  Channel v6;
  Channel v7;
  Channel v8;
  Channel v9;
  Channel v10;
  std::tie(v1, v2) = mainRegion->AddChannel<int64_t>();
  std::tie(v3, v4) = mainRegion->AddChannel<int64_t>();
  std::tie(v5, v6) = mainRegion->AddChannel<int64_t>();
  std::tie(v7, v8) = mainRegion->AddChannel<int64_t>();
  std::tie(v9, v10) = mainRegion->AddChannel<int64_t>();
  mainRegion->AddKpnProcess(v1, v3, v7);
  mainRegion->AddKpnProcess(v2, v4, v5);
  mainRegion->AddKpnProcess(v6, v8, v9);
  mainRegion->AddKpnProcess(v10);
  return;
}


