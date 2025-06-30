pragma solidity ^0.8.0;

contract ContractDemo {
    string message = "Hello Shivam";

    function dispMsg() public view returns (string memory) {
        return message;
    }
}
