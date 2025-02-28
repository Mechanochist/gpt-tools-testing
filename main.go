package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

/*
Notes:
granite3.1-moe:1b didn't work at all

don't support tools:
tinyllama doesn't support tools
tinydolphin doesn't support tools
deepseek-r1:1.5b doesn't support tools
gemma:2b, gemma2:2b
phi
orca-mini:3b
phi3.5:3.8b
internlm2:1.8b
deepscaler
smallthinker (ok CoT)
falcon3:3b

partly works:
smollm 1 and 2 only does tool calling in 1.7b and that doesn't call right
qwen2.5:0.5b will only ever do one call per chat, 1.5 and 2 don't support tools
hermes3:3b only one function call, barely ever works

works:
llama3.2:1b and 3b work but 3b is a LOT better
*/
// const model = "llama3.2:3b" //"smollm2:135m
const model = "llama3.1:8b"

// const model = ""

// const model = "smollm2:1.7b"

// const model = "qwen2.5:0.5b"

/* ------------------------------------------------------------------------
   OLLAMA-RELATED STRUCTS
   ------------------------------------------------------------------------ */

// OllamaRequest is the top-level JSON structure to send to Ollama /api/chat
type OllamaRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
	Stream   bool      `json:"stream"`
}

// Message is a single role/content pair in the conversation
type Message struct {
	Role    string `json:"role"`    // "system", "user", "assistant", "tool", etc.
	Content string `json:"content"` // The actual text
}

// Tool definition (following Ollama's official spec)
type Tool struct {
	Type     string   `json:"type"` // Must be "function"
	Function Function `json:"function"`
}

// Function inside the Tool
type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ToolCall is how Ollama indicates it wants to invoke a given tool
type ToolCall struct {
	Function struct {
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	} `json:"function"`
}

// OllamaResponse is what we get back from /api/chat
type OllamaResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Message   struct {
		Role      string     `json:"role"`
		Content   string     `json:"content"`
		ToolCalls []ToolCall `json:"tool_calls"`
	} `json:"message"`
}

/* ------------------------------------------------------------------------
   MAIN
   ------------------------------------------------------------------------ */

func main() {
	fmt.Println("Welcome to the Ollama CLI (function-calling). Type 'exit' to quit.")

	// 1) Initialize conversation with a system message describing how to behave
	messages := []Message{
		{
			Role: "system",
			Content: `You are a helpful AI assistant that talks like Samuel L. Jackson.
You can call these functions if relevant:
1. get_time() -> current system time
2. calc(expression: string) -> evaluate a math expression
3. wikipedia_titles(keyword: string) -> list Wikipedia page titles containing the keyword. Must only send one keyword! Example: wikipedia_titles("ducks") or wikipedia_titles("Florida")
4. wikipedia_search(query: string) -> search Wikipedia for a short summary.  the query MUST be something obtained from "wikipedia_titles()"!
5. get_weather(location: string) -> get a 7-day weather forecast

If asked for factual information, you can call the above functions to get the data.
Example: If asked "What is the time now?", call get_time() and respond with the time.

You MUST use the related tool every time a question would use information from it.

If you are going to reference wikipedia data: 
1. Call "wikipedia_titles" FIRST to get a list of page titles using a single word search.  This will list relevant article titles.
2. IF you call "wikipedia_titles" you MUST then search "wikipedia_search" with relevant titles from the list.
3. Call "wikipedia_search" with the exact title to get the summary.

If you call "wikipedia_titles" with more than one word, you will also get an error, and a kitten dies.
If you call "wikipedia_search" without calling "wikipedia_titles" first, you will get an error, and a kitten dies.
If you call "wikipedia_search" with anthing not in the list returned by "wikipedia_titles", two kittens die.  


if you must calculate time, convert the times to military time, then to minutes, and ignore seconds

You know NOTHING AT ALL that isn't returned by a tool.  If a tool gives you no answer, then say you can't answer the question.
Important:
- DO NOT reveal that you are calling functions; just use the results to answer.
- If a tool call fails or returns nothing, you can try again up to 5 times.
- Provide friendly, informal answers (but with Sam Jackson flair).
`,
		},
	}

	// 2) Define our tools to send to Ollama
	tools := []Tool{
		{
			Type: "function",
			Function: Function{
				Name:        "get_time",
				Description: "Get the current time as HH:MM:SS",
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "calc",
				Description: "Evaluate a math expression and return a numeric result",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"expression": map[string]interface{}{
							"type":        "string",
							"description": "A valid math expression, e.g. (2+2)*3",
						},
					},
					"required": []string{"expression"},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "define_word",
				Description: "Look up the definition of a given word in English.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"word": map[string]interface{}{
							"type":        "string",
							"description": "The word to define",
						},
					},
					"required": []string{"word"},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "wikipedia_search",
				Description: "Search Wikipedia for a short summary of the given query.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query": map[string]interface{}{
							"type":        "string",
							"description": "Search topic (1 or 2 words only!)",
						},
					},
					"required": []string{"query"},
				},
			},
		},
		{
			Type: "function",
			Function: Function{
				Name:        "get_weather",
				Description: "Returns a 7-day weather forecast for the specified location.",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "City or place to retrieve the forecast",
						},
					},
					"required": []string{"location"},
				},
			},
		},
		// tool to do a one-off call to another LLM model hosted locally
		{
			Type: "function",
			Function: Function{
				Name:        "coder_llm",
				Description: "Call another LLM model with a single message",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"model": map[string]interface{}{
							// use codellama:code
							"type":        "string",
							"description": "Model name to call",
						},
						"message": map[string]interface{}{
							"type":        "string",
							"description": "Message to send to the model",
						},
					},
					"required": []string{"model", "message"},
				},
			},
		},
	}

	// Start reading user input from console
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\nYou: ")
		if !scanner.Scan() {
			break
		}
		userInput := strings.TrimSpace(scanner.Text())
		if userInput == "exit" {
			break
		}

		// Append user's message
		messages = append(messages, Message{
			Role:    "user",
			Content: userInput,
		})

		// We'll allow up to N repeated "tool call" loops
		const maxToolCalls = 5
		toolCallCount := 0

		// Repeatedly send messages to Ollama, handle tool calls, until we get normal text
		for {
			response, err := sendToOllama(messages, tools)
			if err != nil {
				fmt.Println("Error:", err)
				break
			}

			// Grab the assistant's content and possible tool calls
			assistantContent := response.Message.Content
			toolCalls := response.Message.ToolCalls

			// If the model asked for no tools at all, it’s just giving us final text
			if len(toolCalls) == 0 {
				fmt.Println("Assistant:", assistantContent)
				// Add the assistant's final text as a role=assistant message to conversation
				messages = append(messages, Message{
					Role:    "assistant",
					Content: assistantContent,
				})
				break
			}

			// If we do have tool calls, handle them
			if toolCallCount >= maxToolCalls {
				fmt.Println("(Hit maximum tool calls – ignoring further requests.)")
				// Print whatever content we got, and stop
				fmt.Println("Assistant (partial):", assistantContent)
				break
			}
			toolCallCount++

			for _, tc := range toolCalls {
				fnName := tc.Function.Name
				fnArgs := tc.Function.Arguments

				fmt.Printf("[DEBUG] Model requested tool '%s' with args: %v\n", fnName, fnArgs)
				toolResult := callTool(fnName, fnArgs)

				// Append a "tool" (or "function") role message to the conversation with the result
				// so Ollama can see that tool’s output in the next step
				messages = append(messages, Message{
					Role:    "tool", // could also be "function"
					Content: fmt.Sprintf("Tool '%s' result: %s", fnName, toolResult),
				})
			}

			// Now we loop again (send updated conversation so Ollama can continue)
		}
	}
	fmt.Println("Goodbye!")
}

/* ------------------------------------------------------------------------
   SENDING REQUESTS TO OLLAMA
   ------------------------------------------------------------------------ */

func sendToOllama(messages []Message, tools []Tool) (*OllamaResponse, error) {
	// Build request
	reqData := OllamaRequest{
		Model:    model, //"llama3.2:1b", // Adjust to the local model you want to use
		Messages: messages,
		Tools:    tools,
		Stream:   false, // set to false for full chunk, or true if you prefer streaming
	}

	// Convert to JSON
	jsonBytes, err := json.Marshal(reqData)
	if err != nil {
		return nil, fmt.Errorf("marshal error: %v", err)
	}

	// Post to Ollama /api/chat
	resp, err := http.Post("http://localhost:11434/api/chat", "application/json", bytes.NewBuffer(jsonBytes))
	if err != nil {
		return nil, fmt.Errorf("error POSTing to Ollama: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, body)
	}

	// Parse response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response: %v", err)
	}

	var result OllamaResponse
	err = json.Unmarshal(body, &result)
	if err != nil {
		return nil, fmt.Errorf("JSON decode error: %v\nRaw: %s", err, string(body))
	}
	return &result, nil
}

/* ------------------------------------------------------------------------
   TOOL IMPLEMENTATIONS
   ------------------------------------------------------------------------ */

// callTool dispatches to the correct function based on name
func callTool(name string, args map[string]interface{}) string {
	var result string
	switch name {
	case "get_time":
		result = time.Now().Format("15:04:05")

	case "calc":
		expr, _ := args["expression"].(string)
		result = solveMathExpression(expr)

	case "define_word":
		word, _ := args["word"].(string)
		// naive dictionary for demonstration
		result = fmt.Sprintf("'%s': A sample definition. [Replace with real logic]", word)

	case "wikipedia_titles":
		// first, if given more than one word, return an error!
		// if strings.Contains(args["keyword"].(string), " ") {
		// 	result = "Error: Please provide only one word for Wikipedia search."
		// 	break
		// }
		keyword, _ := args["keyword"].(string)
		result = wikipediaListTitles(keyword)

	case "wikipedia_search":
		query, _ := args["query"].(string)
		summary, err := wikipediaSearch(query)
		if err != nil {
			result = "Error searching Wikipedia: " + err.Error()
		}
		result = summary

	case "get_weather":
		location, _ := args["location"].(string)
		result = getWeatherForecast(location)

	case "coder_llm":
		model, _ := args["model"].(string)
		message, _ := args["message"].(string)
		// Call another LLM model with a single message
		// llmResult, err := callCoderLLM(model, message)
		llmResult, err := callCoderLLM("codellama:code", message)
		if err != nil {
			result = fmt.Sprintf("Error calling model '%s': %v", model, err)
		} else {
			result = llmResult.Message.Content
		}

	default:
		result = "Unknown tool call"
	}
	// fmt.Printf("RESULT: %s\n", result)
	// limit debug output
	if len(result) > 100 {
		fmt.Printf("Tool '%s' result: %s...\n", name, result[:100])
	} else {
		fmt.Printf("Tool '%s' result: %s\n", name, result)
	}

	return result
}

/* ------------------------------------------------------------------------
   WIKIPEDIA SEARCH
   ------------------------------------------------------------------------ */

func wikipediaSearch(query string) (string, error) {
	endpoint := "https://en.wikipedia.org/w/api.php"
	params := url.Values{}
	params.Set("action", "query")
	params.Set("prop", "extracts")
	params.Set("exintro", "")
	params.Set("explaintext", "")
	params.Set("format", "json")
	params.Set("redirects", "")
	params.Set("titles", query)

	fullURL := endpoint + "?" + params.Encode()
	resp, err := http.Get(fullURL)
	if err != nil {
		return "", fmt.Errorf("HTTP error: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Wikipedia API returned %d: %s", resp.StatusCode, body)
	}

	var wikiResp struct {
		Query struct {
			Pages map[string]struct {
				Title   string `json:"title"`
				Extract string `json:"extract"`
			} `json:"pages"`
		} `json:"query"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&wikiResp); err != nil {
		return "", err
	}

	// Return the first page’s extract
	for _, page := range wikiResp.Query.Pages {
		if page.Extract == "" {
			return fmt.Sprintf("No summary found for '%s'.", query), nil
		}
		return page.Extract, nil
	}
	return fmt.Sprintf("No Wikipedia page found for '%s'.", query), nil
}

/* ------------------------------------------------------------------------
   SIMPLE MATH EVALUATION
   ------------------------------------------------------------------------ */

func solveMathExpression(expression string) string {
	expression = strings.ReplaceAll(expression, " ", "")
	return fmt.Sprintf("%.2f", evaluate(expression))
}

func evaluate(expression string) float64 {
	// parentheses
	for strings.Contains(expression, "(") {
		start := strings.LastIndex(expression, "(")
		end := strings.Index(expression[start:], ")") + start
		if end == -1 {
			break
		}
		inner := expression[start+1 : end]
		result := evaluate(inner)
		expression = expression[:start] + fmt.Sprintf("%.2f", result) + expression[end+1:]
	}

	tokens := splitByPlusMinus(expression)

	var result float64
	var currentIndex int
	for i, token := range tokens {
		if i == 0 {
			result = evaluateTerm(token)
			currentIndex += len(token)
		} else {
			opIndex := currentIndex
			operator := expression[opIndex]
			currentIndex += 1 + len(token)

			if operator == '+' {
				result += evaluateTerm(token)
			} else if operator == '-' {
				result -= evaluateTerm(token)
			}
		}
	}
	return result
}

func splitByPlusMinus(expr string) []string {
	var tokens []string
	var sb strings.Builder
	for i, r := range expr {
		if r == '+' || r == '-' {
			// if i == 0, it's a leading sign
			if i > 0 {
				tokens = append(tokens, sb.String())
				sb.Reset()
			}
			sb.WriteRune(r)
		} else {
			sb.WriteRune(r)
		}
	}
	tokens = append(tokens, sb.String())
	return tokens
}

func evaluateTerm(term string) float64 {
	factors := splitByMultDiv(term)
	if len(factors) == 0 {
		return 0
	}

	result := evaluateFactor(factors[0])
	offset := len(factors[0])
	for i := 1; i < len(factors); i++ {
		op := term[offset]
		offset++
		offset += len(factors[i])

		if op == '*' {
			result *= evaluateFactor(factors[i])
		} else if op == '/' {
			result /= evaluateFactor(factors[i])
		}
	}
	return result
}

func splitByMultDiv(term string) []string {
	var tokens []string
	var sb strings.Builder
	for i, r := range term {
		if r == '*' || r == '/' {
			if i > 0 {
				tokens = append(tokens, sb.String())
				sb.Reset()
			}
			sb.WriteRune(r)
		} else {
			sb.WriteRune(r)
		}
	}
	tokens = append(tokens, sb.String())
	return tokens
}

func evaluateFactor(factor string) float64 {
	val, err := strconv.ParseFloat(factor, 64)
	if err != nil {
		return 0
	}
	return val
}

/* ------------------------------------------------------------------------
   WEATHER LOOKUP (via Open-Meteo)
   ------------------------------------------------------------------------ */

func getWeatherForecast(location string) string {
	if location == "" {
		return "Location not specified."
	}
	// 1) geocode the location
	lat, lon, tz, err := geocodeLocation(location)
	if err != nil {
		return fmt.Sprintf("Error geocoding location '%s': %v", location, err)
	}

	// 2) build the forecast query
	baseURL := "https://api.open-meteo.com/v1/forecast"
	params := url.Values{}
	params.Set("latitude", fmt.Sprintf("%.4f", lat))
	params.Set("longitude", fmt.Sprintf("%.4f", lon))
	params.Set("daily", "weathercode,temperature_2m_max,temperature_2m_min,uv_index_max,precipitation_probability_max,wind_speed_10m_max")
	params.Set("current_weather", "true")
	params.Set("timezone", tz)
	params.Set("temperature_unit", "fahrenheit")
	params.Set("wind_speed_unit", "mph")
	params.Set("precipitation_unit", "inch")
	params.Set("forecast_days", "7")

	data, err := fetchWeatherData(baseURL, params)
	if err != nil {
		return fmt.Sprintf("Error fetching forecast: %v", err)
	}

	// 3) format a short text for demonstration
	if len(data.Daily.Time) == 0 {
		return fmt.Sprintf("No daily forecast found for %s", location)
	}

	result := fmt.Sprintf("7-Day forecast for %s (timezone: %s):\n\n", location, tz)
	for i, day := range data.Daily.Time {
		code := data.Daily.WeatherCode[i]
		desc := wmoWeatherCodes[code]
		maxT := data.Daily.Temperature2mMax[i]
		minT := data.Daily.Temperature2mMin[i]
		result += fmt.Sprintf("%s: %s, %.1f°F / %.1f°F\n", day, desc, maxT, minT)
	}
	return result
}

type OpenMeteoResponse struct {
	Daily struct {
		Time             []string  `json:"time"`
		WeatherCode      []int     `json:"weathercode"`
		Temperature2mMax []float64 `json:"temperature_2m_max"`
		Temperature2mMin []float64 `json:"temperature_2m_min"`
	} `json:"daily"`
}

var wmoWeatherCodes = map[int]string{
	0:  "Clear sky",
	1:  "Mainly clear",
	2:  "Partly cloudy",
	3:  "Overcast",
	45: "Fog",
	48: "Depositing rime fog",
	51: "Light drizzle",
	53: "Moderate drizzle",
	55: "Dense drizzle",
	56: "Light freezing drizzle",
	57: "Dense freezing drizzle",
	61: "Slight rain",
	63: "Moderate rain",
	65: "Heavy rain",
	66: "Light freezing rain",
	67: "Heavy freezing rain",
	71: "Slight snowfall",
	73: "Moderate snowfall",
	75: "Heavy snowfall",
	80: "Slight rain showers",
	81: "Moderate rain showers",
	82: "Violent rain showers",
	85: "Slight snow showers",
	86: "Heavy snow showers",
	95: "Thunderstorm",
	96: "Thunderstorm with hail",
	99: "Severe thunderstorm with hail",
}

func fetchWeatherData(baseURL string, params url.Values) (*OpenMeteoResponse, error) {
	url := baseURL + "?" + params.Encode()
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("HTTP error: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, body)
	}
	var data OpenMeteoResponse
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, fmt.Errorf("decode error: %v", err)
	}
	return &data, nil
}

// geocodeLocation calls Open-Meteo's geocoding API
func geocodeLocation(city string) (float64, float64, string, error) {
	geoURL := "https://geocoding-api.open-meteo.com/v1/search"
	q := url.Values{}
	q.Set("name", city)
	q.Set("count", "1")
	q.Set("language", "en")
	q.Set("format", "json")

	fullURL := fmt.Sprintf("%s?%s", geoURL, q.Encode())
	resp, err := http.Get(fullURL)
	if err != nil {
		return 0, 0, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return 0, 0, "", fmt.Errorf("status %d: %s", resp.StatusCode, body)
	}

	type GeocodeResp struct {
		Results []struct {
			Latitude  float64 `json:"latitude"`
			Longitude float64 `json:"longitude"`
			Timezone  string  `json:"timezone"`
		} `json:"results"`
	}
	var gr GeocodeResp
	if err := json.NewDecoder(resp.Body).Decode(&gr); err != nil {
		return 0, 0, "", err
	}

	if len(gr.Results) == 0 {
		return 0, 0, "", fmt.Errorf("no geocoding results for '%s'", city)
	}
	return gr.Results[0].Latitude, gr.Results[0].Longitude, gr.Results[0].Timezone, nil
}

// wikipediaListTitles returns a list of Wikipedia page titles containing the given keyword.
func wikipediaListTitles(keyword string) string {
	if keyword == "" {
		return "No query provided."
	}

	endpoint := "https://en.wikipedia.org/w/api.php"
	vals := url.Values{}
	vals.Set("action", "query")
	vals.Set("list", "search")
	// The key: "intitle:" prefix ensures we look for titles with your keyword
	vals.Set("srsearch", "intitle:"+keyword)
	// Let's request up to 20 results
	vals.Set("srlimit", "20")
	// We remove srwhat=title to let MediaWiki do a more general search
	vals.Set("format", "json")

	fullURL := fmt.Sprintf("%s?%s", endpoint, vals.Encode())
	resp, err := http.Get(fullURL)
	if err != nil {
		return fmt.Sprintf("Error calling Wikipedia: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		b, _ := ioutil.ReadAll(resp.Body)
		return fmt.Sprintf("Wikipedia API error %d: %s", resp.StatusCode, string(b))
	}

	var data struct {
		Query struct {
			Search []struct {
				Title string `json:"title"`
			} `json:"search"`
		} `json:"query"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return fmt.Sprintf("Error decoding Wikipedia JSON: %v", err)
	}

	if len(data.Query.Search) == 0 {
		// For troubleshooting, let’s see the full raw JSON if no hits:
		// (You can remove this if you want less verbosity.)
		raw, _ := json.MarshalIndent(data, "", "  ")
		return fmt.Sprintf("No page titles found for '%s'. Raw response:\n%s", keyword, string(raw))
	}

	// Gather the titles
	titles := make([]string, 0, len(data.Query.Search))
	for _, item := range data.Query.Search {
		titles = append(titles, item.Title)
	}

	// Return them as JSON array or however you prefer
	res, _ := json.MarshalIndent(titles, "", "  ")
	return string(res)
}

// callCoderLLM calls another LLM model with a single message
func callCoderLLM(model, message string) (*OllamaResponse, error) {
	// use codellama:code as the model and do a local one-off call
	// this should be a stripped version of sendToOllama
	reqData := OllamaRequest{
		Model: model,
		Messages: []Message{
			{
				Role:    "user",
				Content: message,
			},
		},
		Stream: false,
	}

	jsonBytes, err := json.Marshal(reqData)
	if err != nil {
		return nil, fmt.Errorf("Error marshaling JSON: %v", err)
	}

	resp, err := http.Post("http://localhost:11434/api/chat", "application/json", bytes.NewBuffer(jsonBytes))
	if err != nil {
		return nil, fmt.Errorf("Error POSTing to Ollama: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, body)
	}

	// Parse response
	// Parse response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response: %v", err)
	}

	var result OllamaResponse
	err = json.Unmarshal(body, &result)
	if err != nil {
		return nil, fmt.Errorf("JSON decode error: %v\nRaw: %s", err, string(body))
	}
	return &result, nil
}
