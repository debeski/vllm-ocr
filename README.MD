# DeepSeek-OCR + vLLM Local Container | حاوية محلية لقارئ النصوص ديب سيك

<table>
<tr>
<th>English</th>
<th>العربية</th>
</tr>

<tr>
<td colspan="2" align="center"><strong>Description / الوصف</strong></td>
</tr>
<tr>
<td>This repository contains a <strong>DeepSeek-OCR</strong> setup baked inside a <strong>vLLM</strong> container. Running this will give you a fully local AI text recognition system accessible via a <strong>Gradio</strong> interface on <code>localhost:7860</code> and also provides a <strong>Jupyter Notebook</strong> for further exploration and tweaking.</td>
<td dir="rtl" lang="ar">يحتوي هذا المستودع على ملفات إعداد <strong>قارئ النصوص من DeepSeek</strong> مدمج داخل <strong>حاوية vLLM</strong>. عند تشغيله، ستحصل على نظام ذكاء اصطناعي محلي كامل للتعرف على النصوص متاح عبر واجهة <strong>Gradio</strong> على <code>localhost:7860</code>، كما يتوفر <strong>Jupyter Notebook</strong> لمزيد من التخصيص.</td>
</tr>

<tr>
<td colspan="2" align="center"><strong>Warning / تنبيه</strong></td>
</tr>
<tr>
<td>This is a powerful AI agent running locally. It is <strong>hardware intensive</strong>. Ensure your device can handle it and has an NVIDIA GPU stronger than a 2070.</td>
<td dir="rtl" lang="ar">هذا نظام ذكاء اصطناعي قوي يعمل محليًا. يتطلب موارد جهاز عالية. تأكد من أن جهازك قادر على تشغيله وأن لديك بطاقة رسوميات <strong>إنفيديا أقوى من 2070</strong>.</td>
</tr>

<tr>
<td colspan="2" align="center"><strong>Language Support / دعم اللغات</strong></td>
</tr>
<tr>
<td>The OCR works for <strong>all languages</strong>, including Arabic.</td>
<td dir="rtl" lang="ar">قارئ النصوص يعمل على <strong>جميع اللغات</strong> بما في ذلك العربية.</td>
</tr>

<tr>
<td colspan="2" align="center"><strong>Requirements / المتطلبات</strong></td>
</tr>
<tr>
<td>- Docker installed<br>- Latest NVIDIA video drivers</td>
<td dir="rtl" lang="ar">- تثبيت <strong>دوكر</strong><br>- أحدث برامج تشغيل بطاقة الرسوميات <strong>إنفيديا</strong></td>
</tr>

<tr>
<td colspan="2" align="center"><strong>Instructions / الطريقة</strong></td>
</tr>
<tr>
<td>1. Clone the repository:<br><code>git clone https://github.com/debeski/db-deepseek-ocr</code><br><code>cd db-deepseek-ocr</code><br>2. Build the container (downloads & bakes the model inside):<br><code>docker compose build</code><br>3. Run the container:<br><code>docker compose up</code><br>4. Open your browser:<br>- Gradio OCR interface: <code>http://localhost:7860</code><br>- Jupyter Notebook: <code>http://localhost:8888</code></td>
<td dir="rtl" lang="ar">1. استنساخ المستودع:<br><code>git clone https://github.com/debeski/db-deepseek-ocr</code><br><code>cd db-deepseek-ocr</code><br>2. بناء <strong>الحاوية</strong> (سيقوم بتحميل النموذج ودمجه داخلها):<br><code>docker compose build</code><br>3. تشغيل <strong>الحاوية</strong>:<br><code>docker compose up</code><br>4. افتح المتصفح:<br>- واجهة قارئ النصوص: <code>http://localhost:7860</code><br>- دفتر جوبتر: <code>http://localhost:8888</code></td>
</tr>

<tr>
<td colspan="2" align="center"><strong>Licenses / الرخص</strong></td>
</tr>
<tr>
<td>- This project includes <strong>DeepSeek-OCR</strong> and <strong>vLLM</strong>, both open-source. Refer to their repositories for license terms.<br>- The code in this repository is under the <strong>MIT License</strong>, allowing free use, modification, and distribution <strong>without any warranty</strong>.<br>- DeepSeek-OCR License: <a href="https://github.com/deepseek-ai/DeepSeek-OCR">GitHub</a><br>- vLLM License: <a href="https://github.com/vllm-project/vllm">GitHub</a></td>
<td dir="rtl" lang="ar">- هذا المشروع يتضمن <strong>قارئ النصوص DeepSeek-OCR</strong> و <strong>vLLM</strong>، وكلاهما مفتوح المصدر. يرجى مراجعة مستودعاتهم الأصلية لشروط الرخص.<br>- الشيفرة في هذا المستودع متاحة بموجب <strong>رخصة MIT</strong>، مما يسمح باستخدامها وتعديلها وتوزيعها بحرية، <strong>بدون أي ضمان</strong>.<br>- رخصة DeepSeek-OCR: <a href="https://github.com/deepseek-ai/DeepSeek-OCR">مستودع GitHub</a><br>- رخصة vLLM: <a href="https://github.com/vllm-project/vllm">مستودع GitHub</a></td>
</tr>
</table>