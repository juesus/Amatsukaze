/**
* Amatsukaze common header
* Copyright (c) 2017 Nekopanda
*
* This software is released under the MIT License.
* http://opensource.org/licenses/mit-license.php
*/
#pragma once

// ターゲットを Windows Vista に設定
#define _WIN32_WINNT 0x0600 // _WIN32_WINNT_VISTA
#include <SDKDDKVer.h>

#define WIN32_LEAN_AND_MEAN             // Windows ヘッダーから使用されていない部分を除外します。
// Windows ヘッダー ファイル:
#include <windows.h>

// min, maxマクロは必要ないので削除
#undef min
#undef max

#include <cstdint>
#include <stdio.h>

#define PRINTF(...) printf(__VA_ARGS__); fflush(stdout)

inline void assertion_failed(const char* line, const char* file, int lineNum) {
	char buf[500];
	sprintf_s(buf, "Assertion failed!! %s (%s:%d)", line, file, lineNum);
	PRINTF("%s\n", buf);
	//MessageBox(NULL, "Error", "Amatsukaze", MB_OK);
	throw buf;
}

#ifndef _DEBUG
#define ASSERT(exp)
#else
#define ASSERT(exp) do { if(!(exp)) assertion_failed(#exp, __FILE__, __LINE__); } while(0)
#endif

inline int __builtin_clzl(uint64_t mask) {
	DWORD index;
#ifdef _WIN64
	_BitScanReverse64(&index, mask);
#else
	DWORD highWord = (DWORD)(mask >> 32);
	DWORD lowWord = (DWORD)mask;
	if (highWord) {
		_BitScanReverse(&index, highWord);
		index += 32;
	}
	else {
		_BitScanReverse(&index, lowWord);
	}
#endif
	return index;
}
