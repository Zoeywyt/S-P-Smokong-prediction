一、项目概述
本项目是一款集成数学公式 OCR 识别、中英文混合文本 OCR 识别、日历转换与计算、进制转换、长度转换以及计算记录管理等多功能的 Android 应用程序。通过 Tesseract OCR 引擎实现图像识别，结合多种数据处理技术，提供全面的数学与文本处理能力。应用采用模块化设计，界面友好，支持多线程处理以提升用户体验。
二、核心功能
1. OCR 识别模块
公式 OCR：支持手写 / 印刷公式识别，通过拉普拉斯边缘检测、自适应二值化等预处理流程提升识别精度，后处理规则实现符号标准化（如 × 转 *）和公式类型适配（三角函数 / 微积分 / 代数）。
文本 OCR：支持中英混合文本识别，优化中文分割（合并 “中 文” 为 “中文”），兼容多语言模型（chi_sim+eng）。
2. 计算与转换模块
基础计算：基于 math.js 引擎实现复杂表达式计算，支持异步回调和错误分类（ANS/ERROR/FUNCTION）。
单位转换：采用 Map 数据结构存储单位换算系数，使用 BigDecimal 避免浮点误差，支持长度、面积、体积等多类型单位。
进制转换：基于 Integer.parseInt 算法实现 2/8/10/16 进制转换，支持带小数数值，处理 60 进制时间格式特殊场景。
日历转换：1900-2100 年公历 / 农历双向转换，支持生肖计算（(年份 - 4)%12 映射）和日期间隔计算。
3. 用户体验优化
动态键盘：通过 JSON 配置键盘布局，支持长按编辑（MaterialDialog 交互）和多模式切换（进制 / 单位面板动态显示）。
历史记录：存储计算、进制转换、单位转换历史，支持清除和持久化（文件序列化）。
三、技术架构
plaintext
UI层（Activity/Fragment）
├─ 业务逻辑层（模块化功能组件）
│  ├─ OCR处理（FormulaOcrProcessor/TextOcrActivity）
│  ├─ 计算引擎（MathParser/math.js）
│  └─ 转换工具（单位/进制/日历转换器）
├─ 核心引擎（Tesseract OCR/math.js）
└─ 数据层（文件存储/序列化历史记录）
四、关键技术实现
1. OCR 引擎集成
初始化：配置字符白名单（包含数学符号如∂∇∫），设置页面分割模式（PSM_SINGLE_BLOCK），加载训练数据（equ.traineddata/eng.traineddata）。
预处理：
公式：灰度化→对比度增强→拉普拉斯边缘检测→自适应二值化→降噪→缩放。
文本：温和对比度调整→中文优化阈值二值化→等比例缩放。
后处理：
公式：符号标准化（×→*）、类型特定校正（如 sin 后自动补括号）。
文本：合并中文空格、中英文混排空格优化。
2. 高精度计算
单位换算：全程使用 BigDecimal，保留 6 位小数，四舍五入处理。
进制转换：支持小数转换，通过 parseNumber/formatNumber 方法处理整数与小数部分，捕获非法字符异常。
3. 性能优化
多线程处理：OCR 预处理使用 ExecutorService，避免阻塞主线程。
内存管理：及时释放 Tesseract 引擎资源（tessBaseAPI.end ()），复用 Bitmap 对象。
数据存储：延迟加载非关键数据，使用文件序列化存储历史记录。
五、开发里程碑
阶段	Q1 2023	Q2 2023	Q3 2023
核心计算	■■■■		
OCR 集成	■■		■■■■
性能优化		■■	
六、未来规划
1. 技术演进
OCR 增强：集成 OpenCV 手写检测，支持 LaTeX 公式输出。
AI 集成：接入 GPT-4 实现数学解释，拍照生成解题步骤。
2. 产品扩展
多语言支持：新增日语 / 韩语 OCR 识别。
可视化功能：支持函数图形 3D 渲染。
七、代码结构
plaintext
calcaulator2/
├─ 核心类（OCR/计算/转换）
│  ├─ FormulaOcrProcessor.java   # 公式OCR核心处理
│  ├─ MathParser.java            # 表达式解析器
│  └─ LunarSolarConverter.java   # 农历/公历转换工具
├─ 功能界面
│  ├─ FormulaOcrActivity.java    # 公式OCR界面
│  ├─ TextOcrActivity.java       # 文本OCR界面
│  ├─ calcaulateActivity.java    # 计算主界面
│  └─ CalendarConverterActivity.java # 日历转换界面
├─ 工具类
│  └─ Pix2TextIntegration.java   # 图像转文本集成
└─ 数据模型
   └─ MathItem.java              # 计算记录模型
八、使用说明
环境要求：Android Studio 4.0+，API 23+，Tesseract 训练数据需放置于 assets/tessdata 目录。
关键依赖：
Tesseract OCR 引擎（com.rmtheis:tess-two:9.1.0）
math.js（通过 WebView 集成）
运行步骤：
克隆仓库至本地，导入 Android Studio
确保训练数据文件正确复制
编译运行（建议使用 API 30 + 模拟器测试）
九、贡献指南
欢迎提交 PR 或 Issue，优先关注以下方向：

OCR 识别准确率优化（尤其是手写公式）
性能优化（内存 / 响应速度）
