/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { GoogleGenAI, GenerateContentResponse } from '@google/genai';

// Ensure API_KEY is used as per guidelines
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
const model = 'gemini-2.5-flash-preview-04-17';

const chatMessages = document.getElementById('chat-messages') as HTMLDivElement;
const messageInput = document.getElementById('message-input') as HTMLInputElement;
const sendButton = document.getElementById('send-button') as HTMLButtonElement;
const loadingIndicator = document.getElementById('loading-indicator') as HTMLDivElement;

let strategyData: any = null; // Renamed for clarity

async function loadStrategyData() {
    try {
        const response = await fetch('data_dt_v03.json'); // Changed to data_dt_v03.json
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        strategyData = await response.json();
    } catch (error) {
        console.error('Failed to load data_dt_v03.json:', error);
        addMessageToChat('عذراً، حدث خطأ أثناء تحميل بيانات الاستراتيجية. لا يمكنني المساعدة بدونها.', 'bot');
        disableInput(true); // Disable input if data loading fails
    }
}

function addMessageToChat(text: string, sender: 'user' | 'bot', streamElement?: HTMLDivElement): HTMLDivElement {
    let messageElement = streamElement;

    if (!messageElement) {
        messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        chatMessages.appendChild(messageElement);
    }

    const textNode = document.createTextNode(text);
    messageElement.appendChild(textNode);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageElement;
}


function disableInput(disabled: boolean) {
    messageInput.disabled = disabled;
    sendButton.disabled = disabled;
    loadingIndicator.style.display = disabled ? 'block' : 'none';
}

async function handleSendMessage() {
    const userMessage = messageInput.value.trim();
    if (!userMessage) return;

    addMessageToChat(userMessage, 'user');
    messageInput.value = '';
    disableInput(true);

    if (!strategyData || !strategyData.digitalTransformationStrategy) { // Check for the main strategy object
        addMessageToChat('عذراً، بيانات الاستراتيجية غير متاحة أو بتنسيق غير صحيح. لا يمكنني الإجابة على سؤالك.', 'bot');
        disableInput(false); 
        return;
    }

    const prompt = `أنت مساعد متخصص في الإجابة على الأسئلة المتعلقة باستراتيجية التحول الرقمي الموضحة في بيانات JSON التالية. يجب أن تكون إجاباتك باللغة العربية ومبنية حصريًا على المعلومات الموجودة في هذه البيانات. لا تستخدم أي معرفة خارجية أو تختلق معلومات. إذا لم تتمكن من العثور على إجابة ضمن البيانات المقدمة، أجب بـ "عذراً، لم أتمكن من العثور على إجابة لهذا السؤال في الوثيقة."

بيانات JSON:
${JSON.stringify(strategyData.digitalTransformationStrategy)}

السؤال: ${userMessage}`;

    let botMessageElement = document.createElement('div');
    botMessageElement.classList.add('message', 'bot-message');
    chatMessages.appendChild(botMessageElement); 

    try {
        const responseStream = await ai.models.generateContentStream({
            model: model,
            // contents: [{ role: "user", parts: [{ text: prompt }] }], // Old structure
            contents: prompt, // Simplified content structure as per latest guidelines
        });

        for await (const chunk of responseStream) {
            if (chunk && chunk.text) {
                botMessageElement.textContent += chunk.text;
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
        if (!botMessageElement.textContent?.trim()) {
             botMessageElement.textContent = "لم أتلق ردًا واضحًا من النموذج.";
        }

    } catch (error) {
        console.error('Error calling Gemini API:', error);
        botMessageElement.textContent = 'عذراً، حدث خطأ أثناء محاولة الحصول على إجابة. يرجى المحاولة مرة أخرى.';
    } finally {
        disableInput(false);
        messageInput.focus();
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

sendButton.addEventListener('click', handleSendMessage);
messageInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        handleSendMessage();
    }
});

// Initialize
async function initializeApp() {
    await loadStrategyData();
    if (strategyData && strategyData.digitalTransformationStrategy) {
      addMessageToChat('مرحباً! كيف يمكنني مساعدتك اليوم بناءً على وثيقة استراتيجية التحول الرقمي؟', 'bot');
    } else {
      addMessageToChat('مرحباً! يبدو أن هناك مشكلة في تحميل بيانات الاستراتيجية. يرجى التحقق من وحدة التحكم لمزيد من التفاصيل.', 'bot');
    }
    messageInput.focus();
}

initializeApp();