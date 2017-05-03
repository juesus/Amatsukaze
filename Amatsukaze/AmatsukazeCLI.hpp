/**
* Amtasukaze Command Line Interface
* Copyright (c) 2017 Nekopanda
*
* This software is released under the MIT License.
* http://opensource.org/licenses/mit-license.php
*/
#pragma once

#include <time.h>

#include "TranscodeManager.hpp"

// MSVC�̃}���`�o�C�g��Unicode�łȂ��̂ŕ����񑀍�ɓK���Ȃ��̂�wchar_t�ŕ����񑀍������
#ifdef _MSC_VER
namespace std { typedef wstring tstring; }
typedef wchar_t tchar;
#define stscanf swscanf_s
#define PRITSTR "ls"
#define _T(s) L ## s
#else
namespace std { typedef string tstring; }
typedef char tchar;
#define stscanf sscanf
#define PRITSTR "s"
#define _T(s) s
#endif

static std::string to_string(std::wstring str) {
	if (str.size() == 0) {
		return std::string();
	}
	int dstlen = WideCharToMultiByte(
		CP_ACP, 0, str.c_str(), (int)str.size(), NULL, 0, NULL, NULL);
	std::vector<char> ret(dstlen);
	WideCharToMultiByte(CP_ACP, 0,
		str.c_str(), (int)str.size(), ret.data(), (int)ret.size(), NULL, NULL);
	return std::string(ret.begin(), ret.end());
}

static std::string to_string(std::string str) {
	return str;
}

static void printCopyright() {
	PRINTF(
		"Amatsukaze - Automated MPEG2-TS Transcode Utility\n"
		"Built on %s %s\n"
		"Copyright (c) 2017 Nekopanda\n", __DATE__, __TIME__);
}

static void printHelp(const tchar* bin) {
  PRINTF(
    "%" PRITSTR " <�I�v�V����> -i <input.ts> -o <output.mp4>\n"
    "�I�v�V���� []�̓f�t�H���g�l \n"
    "  -i|--input  <�p�X>  ���̓t�@�C���p�X\n"
    "  -o|--output <�p�X>  �o�̓t�@�C���p�X\n"
		"  -s|--serviceid <���l> ��������T�[�r�XID���w��[]\n"
		"  -w|--work   <�p�X>  �ꎞ�t�@�C���p�X[./]\n"
		"  -et|--encoder-type <�^�C�v>  �g�p�G���R�[�_�^�C�v[x264]\n"
		"                      �Ή��G���R�[�_: x264,x265,QSVEnc,NVEnc\n"
		"  -e|--encoder <�p�X> �G���R�[�_�p�X[x264.exe]\n"
		"  -eo|--encoder-opotion <�I�v�V����> �G���R�[�_�֓n���I�v�V����[]\n"
		"                      ���̓t�@�C���̉𑜓x�A�A�X�y�N�g��A�C���^���[�X�t���O�A\n"
		"                      �t���[�����[�g�A�J���[�}�g���N�X���͎����Œǉ������̂ŕs�v\n"
    "  -b|--bitrate a:b:f  �r�b�g���[�g�v�Z�� �f���r�b�g���[�gkbps = f*(a*s+b)\n"
    "                      s�͓��͉f���r�b�g���[�g�Af�͓��͂�H264�̏ꍇ�͓��͂��ꂽf�����A\n"
    "                      ���͂�MPEG2�̏ꍇ��f=1�Ƃ���\n"
    "                      �w�肪�Ȃ��ꍇ�̓r�b�g���[�g�I�v�V������ǉ����Ȃ�\n"
		"  --2pass             2pass�G���R�[�h\n"
		"  --pulldown          �\�t�g�e���V�l���������Ȃ��ł��̂܂܃G���R�[�h\n"
		"                      �G���R�[�_��--pdfile-in�I�v�V�����ւ̑Ή����K�v\n"
		"  -m|--muxer  <�p�X>  L-SMASH��muxer�ւ̃p�X[muxer.exe]\n"
		"  -t|--timelineeditor <�p�X> L-SMASH��timelineeditor�ւ̃p�X[timelineeditor.exe]\n"
		"  --mpeg2decoder <�f�R�[�_>  MPEG2�p�f�R�[�_[default]\n"
		"                      �g�p�\�f�R�[�_: default,QSV,CUVID"
		"  --h264decoder <�f�R�[�_>  H264�p�f�R�[�_[default]\n"
		"                      �g�p�\�f�R�[�_: default,QSV,CUVID"
		"  -j|--json   <�p�X>  �o�͌��ʏ���JSON�o�͂���ꍇ�͏o�̓t�@�C���p�X���w��[]\n"
    "  --mode <���[�h>     �������[�h[ts]\n"
    "                      ts : MPGE2-TS����͂���ڍ׉�̓��[�h\n"
    "                      g  : MPEG2-TS�ȊO�̓��̓t�@�C�������������ʃt�@�C�����[�h\n"
		"                           ��ʃt�@�C�����[�h��FFMPEG�Ńf�R�[�h���邽�߉��Y���␳\n"
		"                           �Ȃǂ̏����͈�؂Ȃ��̂ŁAMPEG2-TS�ɂ͎g�p���Ȃ��悤��\n"
		"  --dump              �����r���̃f�[�^���_���v�i�f�o�b�O�p�j\n",
		bin);
}

static std::tstring getParam(int argc, tchar* argv[], int ikey) {
	if (ikey + 1 >= argc) {
		THROWF(FormatException,
			"%" PRITSTR "�I�v�V�����̓p�����[�^���K�v�ł�", argv[ikey]);
	}
	return argv[ikey + 1];
}

static std::tstring pathNormalize(std::tstring path) {
	if (path.size() != 0) {
		// �o�b�N�X���b�V���̓X���b�V���ɕϊ�
		std::replace(path.begin(), path.end(), _T('\\'), _T('/'));
		// �Ō�̃X���b�V���͎��
		if (path.back() == _T('/')) {
			path.pop_back();
		}
	}
	return path;
}

template <typename STR>
static size_t pathGetExtensionSplitPos(const STR& path) {
	size_t lastsplit = path.rfind(_T('/'));
	size_t namebegin = (lastsplit == STR::npos)
		? 0
		: lastsplit + 1;
	size_t dotpos = path.find(_T('.'), namebegin);
	size_t len = (dotpos == STR::npos)
		? path.size()
		: dotpos;
	return len;
}

static std::tstring pathRemoveExtension(const std::tstring& path) {
	return path.substr(0, pathGetExtensionSplitPos(path));
}

static std::string pathGetExtension(const std::string& path) {
	auto ext = path.substr(pathGetExtensionSplitPos(path));
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	return ext;
}

static ENUM_ENCODER encoderFtomString(const std::tstring& str) {
	if (str == _T("x264")) {
		return ENCODER_X264;
	}
	else if (str == _T("x265")) {
		return ENCODER_X265;
	}
	else if (str == _T("qsv") || str == _T("QSVEnc")) {
		return ENCODER_QSVENC;
	}
	else if (str == _T("nvenc") || str == _T("NVEnc")) {
		return ENCODER_NVENC;
	}
	else if (str == _T("ffmpeg")) {
		return ENCODER_FFMPEG;
	}
	return (ENUM_ENCODER)-1;
}

static DECODER_TYPE decoderFromString(const std::tstring& str) {
	if (str == _T("default")) {
		return DECODER_DEFAULT;
	}
	else if (str == _T("qsv") || str == _T("QSV")) {
		return DECODER_QSV;
	}
	else if (str == _T("cuvid") || str == _T("CUVID") || str == _T("nvdec") || str == _T("NVDec")) {
		return DECODER_CUVID;
	}
	return (DECODER_TYPE)-1;
}

static std::unique_ptr<TranscoderSetting> parseArgs(AMTContext& ctx, int argc, tchar* argv[])
{
	std::tstring srcFilePath;
	std::tstring outVideoPath;
	std::tstring workDir = _T("./");
	std::tstring outInfoJsonPath;
	ENUM_ENCODER encoder = ENUM_ENCODER();
	std::tstring encoderPath = _T("x264.exe");
  std::tstring encoderOptions = _T("");
	std::tstring muxerPath = _T("muxer.exe");
	std::tstring timelineditorPath = _T("timelineeditor.exe");
  AMT_CLI_MODE mode = AMT_CLI_TS;
  bool autoBitrate = bool();
  BitrateSetting bitrate = BitrateSetting();
  bool twoPass = bool();
	bool pulldown = bool();
	int serviceId = -1;
	DECODER_TYPE mpeg2decoder = DECODER_DEFAULT;
	DECODER_TYPE h264decoder = DECODER_DEFAULT;
	bool dumpStreamInfo = bool();

	for (int i = 1; i < argc; ++i) {
		std::tstring key = argv[i];
		if (key == _T("-i") || key == _T("--input")) {
			srcFilePath = pathNormalize(getParam(argc, argv, i++));
		}
		else if (key == _T("-o") || key == _T("--output")) {
			outVideoPath =
				pathRemoveExtension(pathNormalize(getParam(argc, argv, i++)));
		}
    else if (key == _T("--mode")) {
      std::tstring modeStr = getParam(argc, argv, i++);
      if (modeStr == _T("ts")) {
        mode = AMT_CLI_TS;
      }
      else if (modeStr == _T("g")) {
        mode = AMT_CLI_GENERIC;
      }
      else {
        PRINTF("--mode�̎w�肪�Ԉ���Ă��܂�: %" PRITSTR "\n", modeStr.c_str());
      }
    }
		else if (key == _T("-w") || key == _T("--work")) {
			workDir = pathNormalize(getParam(argc, argv, i++));
		}
		else if (key == _T("-et") || key == _T("--encoder-type")) {
			std::tstring arg = getParam(argc, argv, i++);
			encoder = encoderFtomString(arg);
			if (encoder == (ENUM_ENCODER)-1) {
				PRINTF("--encoder-type�̎w�肪�Ԉ���Ă��܂�: %" PRITSTR "\n", arg.c_str());
			}
		}
		else if (key == _T("-e") || key == _T("--encoder")) {
			encoderPath = getParam(argc, argv, i++);
		}
		else if (key == _T("-eo") || key == _T("--encoder-option")) {
			encoderOptions = getParam(argc, argv, i++);
    }
    else if (key == _T("-b") || key == _T("--bitrate")) {
      const auto arg = getParam(argc, argv, i++);
      int ret = stscanf(arg.c_str(), _T("%lf:%lf:%lf:%lf"),
        &bitrate.a, &bitrate.b, &bitrate.h264, &bitrate.h265);
      if (ret < 3) {
        THROWF(ArgumentException, "--bitrate�̎w�肪�Ԉ���Ă��܂�");
      }
      if (ret <= 3) {
        bitrate.h265 = 2;
      }
      autoBitrate = true;
    }
    else if (key == _T("--2pass")) {
      twoPass = true;
    }
		else if (key == _T("--pulldown")) {
			pulldown = true;
		}
		else if (key == _T("-m") || key == _T("--muxer")) {
			muxerPath = getParam(argc, argv, i++);
		}
		else if (key == _T("-t") || key == _T("--timelineeditor")) {
			timelineditorPath = getParam(argc, argv, i++);
		}
		else if (key == _T("-j") || key == _T("--json")) {
			outInfoJsonPath = getParam(argc, argv, i++);
		}
		else if (key == _T("-s") || key == _T("--serivceid")) {
			std::tstring sidstr = getParam(argc, argv, i++);
			if (sidstr.size() > 2 && sidstr.substr(0, 2) == _T("0x")) {
				// 16�i
				serviceId = std::stoi(sidstr.substr(2), NULL, 16);;
			}
			else {
				// 10�i
				serviceId = std::stoi(sidstr);
			}
		}
		else if (key == _T("--mpeg2decoder")) {
			std::tstring arg = getParam(argc, argv, i++);
			mpeg2decoder = decoderFromString(arg);
			if (mpeg2decoder == (DECODER_TYPE)-1) {
				PRINTF("--mpeg2decoder�̎w�肪�Ԉ���Ă��܂�: %" PRITSTR "\n", arg.c_str());
			}
		}
		else if (key == _T("--h264decoder")) {
			std::tstring arg = getParam(argc, argv, i++);
			h264decoder = decoderFromString(arg);
			if (h264decoder == (DECODER_TYPE)-1) {
				PRINTF("--h264decoder�̎w�肪�Ԉ���Ă��܂�: %" PRITSTR "\n", arg.c_str());
			}
		}
		else if (key == _T("--dump")) {
			dumpStreamInfo = true;
		}
		else if (key.size() == 0) {
			continue;
		}
		else {
			THROWF(FormatException, "�s���ȃI�v�V����: %" PRITSTR, argv[i]);
		}
	}

	if (srcFilePath.size() == 0) {
		THROWF(ArgumentException, "���̓t�@�C�����w�肵�Ă�������");
	}
	if (outVideoPath.size() == 0) {
		THROWF(ArgumentException, "�o�̓t�@�C�����w�肵�Ă�������");
	}

	return std::unique_ptr<TranscoderSetting>(new TranscoderSetting(
		ctx,
		to_string(workDir),
		mode,
		to_string(srcFilePath),
		to_string(outVideoPath),
		to_string(outInfoJsonPath),
		encoder,
		to_string(encoderPath),
		to_string(encoderOptions),
		to_string(muxerPath),
		to_string(timelineditorPath),
		twoPass,
		autoBitrate,
		pulldown,
		bitrate,
		serviceId,
		mpeg2decoder,
		h264decoder,
		dumpStreamInfo));
}

static CRITICAL_SECTION g_log_crisec;
static void amatsukaze_av_log_callback(
  void* ptr, int level, const char* fmt, va_list vl)
{
  level &= 0xff;

  if (level > av_log_get_level()) {
    return;
  }

  char buf[1024];
  vsnprintf(buf, sizeof(buf), fmt, vl);
  int len = (int)strlen(buf);
  if (len == 0) {
    return;
  }

  static char* log_levels[] = {
    "panic", "fatal", "error", "warn", "info", "verb", "debug", "trace"
  };

  EnterCriticalSection(&g_log_crisec);
  
  static bool print_prefix = true;
  bool tmp_pp = print_prefix;
  print_prefix = (buf[len - 1] == '\r' || buf[len - 1] == '\n');
  if (tmp_pp) {
    int logtype = level / 8;
    const char* level_str =
      (logtype >= sizeof(log_levels) / sizeof(log_levels[0]))
      ? "unk" : log_levels[logtype];
    printf("FFMPEG [%s] %s", level_str, buf);
  }
  else {
    printf(buf);
  }
  if (print_prefix) {
    fflush(stdout);
  }

  LeaveCriticalSection(&g_log_crisec);
}

static int amatsukazeTranscodeMain(AMTContext& ctx, const TranscoderSetting& setting) {
	try {
    switch (setting.getMode()) {
    case AMT_CLI_TS:
      transcodeMain(ctx, setting);
      break;
    case AMT_CLI_GENERIC:
      transcodeSimpleMain(ctx, setting);
      break;
    }

		return 0;
	}
	catch (Exception e) {
		return 1;
	}
}
