*** Settings ***
Documentation     A resource file with reusable keywords and variables.
...
...               The system specific keywords created here form our own
...               domain specific language. They utilize keywords provided
...               by the imported SeleniumLibrary.
Library           SeleniumLibraryExtension    run_on_failure=Nothing
Library           String


*** Keywords ***
Go To
    [Arguments]    ${url}
    SeleniumLibraryExtension.Go To    ${url}
    # Execute JavaScript    var i = 0;while (i < 10000) {i++; if (document.readyState == "complete") {break;}}
    Wait For Condition    return document.readyState=="complete"

Click Element
    [Arguments]    ${locator}    ${modifier}=False    ${action_chain}=False
    Wait Until Element Is Visible    ${locator}    120 seconds
    Wait Until Page Contains Element    ${locator}    120 seconds
    Wait Until Element Is Enabled    ${locator}    120 seconds
    Set Focus To Element    ${locator}
    SeleniumLibraryExtension.Click Element    ${locator}    ${modifier}    ${action_chain}

Right Click
    [Arguments]    ${locator}
    Wait Until Element Is Visible    ${locator}    120 seconds
    Sleep    5 seconds
    SeleniumLibraryExtension.Right Click Element    ${locator}

Set Focus To Element
    [Arguments]    ${locator}
    Wait Until Element Is Visible    ${locator}    120 seconds
    SeleniumLibraryExtension.Set Focus To Element    ${locator}

Get Actual data from chart
    [Arguments]    ${locator}
    Right Click and Export Data    ${locator}    class:ctx-menu-action-EC

Right Click and Export Data
    [Arguments]    ${locator}    ${locator1}
    Right Click    ${locator}
    Empty Directory    ${TMPDOWNLOADS}
    Click Element    ${locator1}
    ${file}    Wait Until Keyword Succeeds    1 min    2 sec    Check And Return Download Files
    Log    File was successfully downloaded to ${file}

Right Click and Export Data For QlikSense
    [Arguments]    ${locator}    ${locator1}  ${locator2}
    Right Click    ${locator}
    Empty Directory    ${TMPDOWNLOADS}
    Click Element    ${locator1}
    Click Element    ${locator2}
    ${file}    Wait Until Keyword Succeeds    1 min    2 sec    Check And Return Download Files
    Log    File was successfully downloaded to ${file}

Get actual data from text
    [Arguments]    ${locators}
    Get Text Oject Value    ${locators}

Get Text Oject Value
    [Arguments]    ${locators}
    ${locator_list}    Split String    ${locators}    ,
    ${text}    Create List
    FOR    ${locator}    IN    @{locator_list}
        log    ${locator}
        Set Focus To Element    ${locator}
        ${v}    SeleniumLibraryExtension.Get Text    ${locator}
        Append To List    ${text}    ${v}

    END
    Set Test Variable    ${textobject_value}    ${text}