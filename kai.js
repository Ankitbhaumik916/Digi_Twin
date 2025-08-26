import fs from 'fs';
import ollama from 'ollama';

// Load persona.json
const persona = JSON.parse(fs.readFileSync('./persona.json', 'utf8'));

(async () => {
  console.log("Sending request to Ollama...");

  try {
    const response = await ollama.chat({
      model: 'llama3', // change to your installed model
      messages: [
        { role: 'system', content: `You are ${persona.name}. ${persona.description}` },
        { role: 'user', content: 'Hello, introduce yourself in detail.' }
      ]
    });

    console.log("Ollama replied:\n", response.message?.content || "(No content received)");
  } catch (err) {
    console.error("Error talking to Ollama:", err);
  }
})();
