package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// AWS Credentials (Replace with secure storage in production)
const (
	AWS_ACCESS_KEY    = "..."
	AWS_SECRET_KEY    = "....."
	AWS_SESSION_TOKEN = ".........."
	AWS_REGION        = "us-east-1"
	AWS_MODEL_ID      = "us.meta.llama3-2-90b-instruct-v1:0" // Use the correct Llama3 model ID

	// clickhouseURL      = "https://webhook.site/317cca19-4aa5-4d9c-9c01-2dc348b6b29b/"
	clickhouseURL      = "..."
	clickhouseUser     = "..." // Set your ClickHouse username
	clickhousePassword = "..." // Set your ClickHouse password
	requestTimeout     = 10
)

const systemPrompt = `You are a friendly AI Assistant.  
You can converse normally, as well as call tools when necessary.
You have access to the following tool:
1. "get_time" - Returns the current system time in HH:MM:SS format.
2. "clickhouse_tool" - Executes an SQL query on ClickHouse and returns the response as a JSON string.
2. "no_tool" - don't call a tool (this is a stub).

You do not need to call any tool unless the user SPECIFICALLY requests data that would require the tool, like asking the current time.
In fact, don't mention the tools at all, assume the user knows whatever they need to know to use the tool.`

// Function to get the current time
func getTime() string {
	return time.Now().Format("15:04:05")
}

// ClickhouseTool executes an SQL query on ClickHouse and returns the response as a JSON string.
func clickhouseTool(args ...interface{}) (string, error) {
	// Validate input
	if len(args) == 0 {
		return "", errors.New("missing SQL query argument")
	}
	fmt.Printf("clickhouseTool: args: %v\n", args)
	query, ok := args[0].(string)
	if !ok {
		return "", errors.New("first argument must be a string containing the SQL query")
	}

	// Ensure query is properly formatted
	query = strings.TrimSpace(query)
	if strings.HasSuffix(query, ";") {
		query = query[:len(query)-1]
	}
	query += " FORMAT JSONObjectEachRow;"

	// Prepare request
	req, err := http.NewRequest("POST", clickhouseURL, bytes.NewBuffer([]byte(query)))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}

	// Set headers
	req.Header.Set("X-ClickHouse-User", clickhouseUser)
	req.Header.Set("X-ClickHouse-Key", clickhousePassword)
	req.Header.Set("Content-Type", "text/plain")

	// Execute request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("API request failed: %v", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %v", err)
	}

	// Parse response into JSON
	var result interface{}
	//debugging
	fmt.Println("clickhouseTool: body: ", string(body))
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("error parsing JSON: %v", err)
	}

	// Convert structured response to JSON string
	responseJSON, err := json.MarshalIndent(map[string]interface{}{
		"query":  query,
		"result": result,
	}, "", "  ")
	if err != nil {
		return "", fmt.Errorf("error encoding JSON: %v", err)
	}

	return string(responseJSON), nil
}

func main() {
	// Load AWS Config with Hardcoded Credentials
	cfg, err := config.LoadDefaultConfig(context.TODO(),
		config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_SESSION_TOKEN)),
		config.WithRegion(AWS_REGION),
	)
	if err != nil {
		fmt.Println("Error loading AWS config:", err)
		return
	}

	// Initialize AWS Bedrock client
	client := bedrockruntime.NewFromConfig(cfg)

	// Conversation history
	var conversationHistory []types.Message

	// Tool configuration for get_time
	toolConfig := &types.ToolConfiguration{
		Tools: []types.Tool{
			&types.ToolMemberToolSpec{
				Value: types.ToolSpecification{
					Name:        aws.String("get_time"),
					Description: aws.String("Returns the current system time in HH:MM:SS format."),
					InputSchema: &types.ToolInputSchemaMemberJson{
						Value: document.NewLazyDocument(map[string]interface{}{
							"type":       "object",
							"properties": map[string]interface{}{}, // No input parameters
						}),
					},
				},
			},
			&types.ToolMemberToolSpec{
				Value: types.ToolSpecification{
					Name:        aws.String("clickhouse_tool"),
					Description: aws.String("Executes an SQL query on ClickHouse and returns the response as a JSON string."),
					InputSchema: &types.ToolInputSchemaMemberJson{
						Value: document.NewLazyDocument(map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								// 		"query": map[string]interface{}{
								"type": "string",
								// 	},
							},
							"required": []string{"query"},
						}),
					},
				},
			},
			&types.ToolMemberToolSpec{
				Value: types.ToolSpecification{
					Name:        aws.String("no_tool"),
					Description: aws.String("Stub tool that does nothing."),
					InputSchema: &types.ToolInputSchemaMemberJson{
						Value: document.NewLazyDocument(map[string]interface{}{
							"type":       "object",
							"properties": map[string]interface{}{}, // No input parameters
						}),
					},
				},
			},
		},
		ToolChoice: &types.ToolChoiceMemberAuto{},
	}

	// User input scanner
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("Start chatting with the AI (type 'exit' to quit):")

	for {
		// Get user input
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}
		userInput := scanner.Text()
		if strings.ToLower(userInput) == "exit" {
			break
		}

		// Append user input to conversation history
		conversationHistory = append(conversationHistory, types.Message{
			Role:    types.ConversationRoleUser,
			Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: userInput}},
		})

		// Call Converse API
		resp, err := client.Converse(context.TODO(), &bedrockruntime.ConverseInput{
			ModelId:    aws.String(AWS_MODEL_ID),
			Messages:   conversationHistory,
			ToolConfig: toolConfig,
			System: []types.SystemContentBlock{
				&types.SystemContentBlockMemberText{Value: systemPrompt},
			},
		})
		if err != nil {
			fmt.Println("Error:", err)
			continue
		}

		// Check if the model requested tool invocation
		if resp.StopReason == types.StopReasonToolUse {
			// Extract tool use request
			toolUse, ok := resp.Output.(*types.ConverseOutputMemberMessage)
			if !ok || len(toolUse.Value.Content) == 0 {
				fmt.Println("AI: (Unexpected tool request format)")
				continue
			}

			// Assume the model requested "get_time"
			toolName := *toolUse.Value.Content[0].(*types.ContentBlockMemberToolUse).Value.Name
			// fmt.Printf("Tool request: %v\n", *toolUse.Value.Content[0].(*types.ContentBlockMemberToolUse).Value.Name)
			fmt.Printf("Tool request: %v\n", toolName)
			// if *toolUse.Value.Content[0].(*types.ContentBlockMemberToolUse).Value.Name == "get_time" {
			if toolName == "get_time" {
				// Execute the function
				currentTime := getTime()

				// Format tool response as JSON
				toolResponseData, err := json.Marshal(map[string]string{"time": currentTime})
				if err != nil {
					fmt.Println("Error encoding tool response:", err)
					continue
				}

				// Create tool response message
				toolResponse := types.Message{
					Role: types.ConversationRoleUser,
					Content: []types.ContentBlock{
						&types.ContentBlockMemberText{Value: string(toolResponseData)},
					},
				}

				// Append tool response to conversation history
				conversationHistory = append(conversationHistory, toolResponse)

				// Send tool response back to Bedrock for final response generation
				resp, err = client.Converse(context.TODO(), &bedrockruntime.ConverseInput{
					ModelId:  aws.String(AWS_MODEL_ID),
					Messages: conversationHistory,
				})
				if err != nil {
					fmt.Println("Error:", err)
					continue
				}
				//
			} else if toolName == "clickhouse_tool" {
				toolRequest := toolUse.Value.Content[0]
				inputDoc := toolRequest.(*types.ContentBlockMemberToolUse).Value.Input

				fmt.Printf("Tool request: %v\n", inputDoc)
				var queryMap map[string]interface{}
				err := inputDoc.UnmarshalSmithyDocument(&queryMap)
				if err != nil {
					fmt.Println("Error unmarshalling input:", err)
					continue
				}

				result, err := clickhouseTool(queryMap["query"])
				if err != nil {
					fmt.Println("Error calling ClickHouse tool:", err)
					continue
				}

				// fmt.Println("ClickHouse Result:", result)

				// Append AI tool request to history (AI asking for a tool)
				conversationHistory = append(conversationHistory, types.Message{
					Role:    types.ConversationRoleAssistant,
					Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "Calling ClickHouse tool..."}},
				})

				// Append AI tool request to history (AI asking for a tool)
				conversationHistory = append(conversationHistory, types.Message{
					Role:    types.ConversationRoleAssistant,
					Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "Calling ClickHouse tool..."}},
				})

				// Append tool response to conversation history as a user message (fix)
				conversationHistory = append(conversationHistory, types.Message{
					Role: types.ConversationRoleUser, // <- Change role to USER
					Content: []types.ContentBlock{
						&types.ContentBlockMemberText{Value: result},
					},
				})

				// Send tool response back to Bedrock to generate the final response
				resp, err = client.Converse(context.TODO(), &bedrockruntime.ConverseInput{
					ModelId:  aws.String(AWS_MODEL_ID),
					Messages: conversationHistory,
				})
				if err != nil {
					fmt.Println("Error:", err)
					continue
				}

			} else if toolName == "no_tool" {
				// copy of the get_time but with a "no value" instead of the getTime() function call
				// Format tool response as JSON
				toolResponseData, err := json.Marshal(map[string]string{"no_tool": "no response"})
				if err != nil {
					fmt.Println("Error encoding tool response:", err)
					continue
				}

				// Create tool response message
				toolResponse := types.Message{
					Role: types.ConversationRoleUser,
					Content: []types.ContentBlock{
						&types.ContentBlockMemberText{Value: string(toolResponseData)},
					},
				}

				// Append tool response to conversation history
				conversationHistory = append(conversationHistory, toolResponse)

				// Send tool response back to Bedrock for final response generation
				resp, err = client.Converse(context.TODO(), &bedrockruntime.ConverseInput{
					ModelId:  aws.String(AWS_MODEL_ID),
					Messages: conversationHistory,
				})
				if err != nil {
					fmt.Println("Error:", err)
					continue
				}
			}
		}

		// Extract AI's final response
		if output, ok := resp.Output.(*types.ConverseOutputMemberMessage); ok {
			assistantMessage := output.Value
			if len(assistantMessage.Content) > 0 {
				if textBlock, ok := assistantMessage.Content[0].(*types.ContentBlockMemberText); ok {
					fmt.Println("AI:", textBlock.Value)

					// Append AI response to conversation history
					conversationHistory = append(conversationHistory, assistantMessage)
				} else {
					fmt.Println("AI: (No text response received)")
				}
			} else {
				fmt.Println("AI: (Empty response)")
			}
		} else {
			fmt.Println("AI: (Unexpected response type)")
		}
	}

	fmt.Println("Goodbye!")
}
