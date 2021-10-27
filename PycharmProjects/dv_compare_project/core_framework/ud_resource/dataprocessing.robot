*** Settings ***
Documentation     The resource file with reusable keywords and variables that can use for Qliksense Dashboard automation testing.
...
...               The system specific keywords created here form our own
...               domain specific language. They utilize keywords provided
...               by the imported SeleniumLibrary.

Resource          other.robot
Resource          resource.robot
Resource          seleow.robot
Library           SeleniumLibraryExtension    run_on_failure=Nothing

*** Keywords ***

Open Dashboard
    [Arguments]    ${dashboard_url}=${dashboard_url}
    [Documentation]   Open a dashboard(only *Qliksense* supported).
    ...
    ...    Open dashboard::https://your-dashboard-url
    ...    or
    ...    Open dashboard  https://your-dashboard-url
    ...    *Note: Two or more spaces is needed between keyword and arguments.*

    # Set Common Vars
    Set Test Variable    ${dashboard_url}    ${dashboard_url}
    BI7 Dashboard Login    ${dashboard_url}
    # Execute JavaScript    var i = 0;while (i < 10000) {i++; if (document.readyState == "complete") {break;}}
    Wait For Condition    return document.readyState=="complete"
    Log    document.readyState

Click Button/Label
    [Documentation]   Click a button or label
    ...
    ...    Examples:
    ...    Click Button/Label::Search
    [Arguments]    ${text}
    # Switch Window    locator=main
    # reload page
    # go to    ${dashboard_url}
    Click Element    xpath://*[text()='${text}']

Get Target Obj
    [Documentation]   Give the ObjectID of the test target object.
    ...
    ...    Examples:
    ...    Get Target Obj::uQJPtQ     *Comments*: uQJPtQ is a object id of Qliksense dashboard.
    [Arguments]    ${objid}
    Set Test Variable    ${objid}    ${objid}

Get Actual Data By Filter
    [Documentation]   Get Actual Data By Filter
    [Arguments]    ${filters}
    Set Test Variable    ${filters}    ${filters}

Get Expected Data From Query
    [Documentation]   Get Expected Data From Query
    [Arguments]    ${query}
    Set Test Variable    ${query}    ${query}

Get Actual Data From Query
    [Documentation]   Get Actual Data From Query
    [Arguments]    ${query}
    Set Test Variable    ${query2}    ${query}

Expect Actual Data Equal Expected Data
    [Documentation]   Execute the test case.
    @{param_list} =    Create List  ${objid}  ${filters}  ${query}  schema
    ${params} =  join string  ${param_list}  `$`
    Run Keyword And Continue On Failure  Get Actual and Expect data and Compare them  ${params}
    Switch Window    locator=main
    Click Element    id:clearselections

Expect Actual Data Equal Expected Data For DE
    [Documentation]   Execute DE Compare Logic.
    Run Keyword And Continue On Failure  Compare Raw Data With Insight Data



