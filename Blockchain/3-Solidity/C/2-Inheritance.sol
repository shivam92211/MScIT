pragma solidity >=0.8.0;

contract Parent {
    uint256 internal sum;

    function setValue() external {
        uint256 a = 10;
        uint256 b = 20;
        sum = a + b;
    }
}

contract Child is Parent {
    function getValue() external view returns (uint256) {
        return sum;
    }
}

contract Caller {
    Child cc = new Child();

    function testInheritance() public returns (uint256) {
        cc.setValue();
        return cc.getValue();
    }

    function showValue() public view returns (uint256) {
        return cc.getValue();
    }
}
