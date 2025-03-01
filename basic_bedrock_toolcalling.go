package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
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
	AWS_SECRET_KEY    = "..."
	AWS_SESSION_TOKEN = "......."
	AWS_REGION        = "us-east-1"
	AWS_MODEL_ID      = "us.meta.llama3-2-90b-instruct-v1:0" // Use the correct Llama3 model ID
)

const systemPrompt = `You are a friendly AI Assistant.  
You can converse normally, as well as call tools when necessary.
You have access to the following tool:
1. "get_time" - Returns the current system time in HH:MM:SS format.

You do not need to call any tool unless the user SPECIFICALLY requests data that would require the tool, like asking the current time.
In fact, don't mention the tools at all, assume the user knows whatever they need to know to use the tool.`

// Function to get the current time
func getTime() string {
	return time.Now().Format("15:04:05")
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
			// toolRequest := toolUse.Value.Content[0]
			if true {
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
